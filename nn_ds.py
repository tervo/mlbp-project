'''
GPU command:
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_model.py
'''

# from common import GENRES
from keras.callbacks import Callback
from keras.utils import np_utils, to_categorical
from keras.models import Model, model_from_yaml
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Dropout, Activation, LSTM, \
        TimeDistributed, Convolution1D, MaxPooling1D, Conv1D
from sklearn.model_selection import train_test_split
import numpy as np
import random
import cPickle
from optparse import OptionParser
from sys import stderr, argv
import sys, os
import lib.io
import lib.viz
import lib.cl

SEED = 42
N_LAYERS = 3 #3
FILTER_LENGTH = 5 #5
CONV_FILTER_COUNT = 256 #256
CONV_FILTER_STRIDES = 2
LSTM_COUNT = 256 #256
BATCH_SIZE = 32
EPOCH_COUNT = 600
CLASS_COUNT = 10

def predict(model, x):
    x = np.expand_dims(x, axis=2)
    print("x shape in prediction: %d, %d, %d"%x.shape)
    return model.predict(x)
    
def load_model(model_filename, weights_filename):
    with open(model_filename, 'r') as f:
        model = model_from_yaml(f.read())
    model.load_weights(weights_filename)
    # self.pred_fun = get_layer_output_function(model, 'output_realtime')
    print 'Model loaded'
    return model

def train_model(x_train, y_train, x_val, y_val):
    print 'Building model...'

    B = K.backend()
    if B=='tensorflow':
        K.set_image_dim_ordering('tf')
    
    x_train = np.expand_dims(x_train, axis=2)
    x_val = np.expand_dims(x_val, axis=2) 
    print("x_train shape: %d, %d, %d"%x_train.shape)
    #print x_train
    y_cat = to_categorical(y_train)
    y_val = to_categorical(y_val)
    
    n_features = x_train.shape[1]
    
    # input_shape = (n_features, None)
    # input_shape = (None, n_features)
    input_shape = (n_features, 1)

    model_input = Input(input_shape, name='input')
    layer = model_input
    
    for i in range(N_LAYERS):
        layer = Conv1D(
            filters = CONV_FILTER_COUNT,
            kernel_size = FILTER_LENGTH,
            name='convolution_' + str(i + 1)
        )(layer)

        layer = Activation('relu')(layer)
        layer = MaxPooling1D(2)(layer)
        #layer = Dropout(0.25)(layer)

    layer = Dropout(0.5)(layer)
    layer = LSTM(LSTM_COUNT, return_sequences=True)(layer)
    layer = Dropout(0.5)(layer)
    layer = TimeDistributed(Dense(CLASS_COUNT))(layer)
    layer = Activation('softmax', name='output_realtime')(layer)
    time_distributed_merge_layer = Lambda(
            function=lambda x: K.mean(x, axis=1), 
            output_shape=lambda shape: (shape[0],) + shape[2:],
            name='output_merged'
    )
    model_output = time_distributed_merge_layer(layer)
    model = Model(model_input, model_output)
    opt = RMSprop(lr=0.00001)
    #opt = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
        )

    print 'Training...'
    # history = model.fit(x=x_train, y=y_cat, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT,
    # validation_data=(x_val, y_val), verbose=2)

    return model

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-s', '--save_path', dest='model_path',
                      default=os.path.join(os.path.dirname(__file__),
                    'models'),
                      help='path to the output model', metavar='MODEL_PATH')
    parser.add_option('-l', '--load_path', dest='load_path',
                      default=None,
                      help='path to the load model',
                      metavar='LOAD_PATH')
    parser.add_option('-o', '--output_path', dest='output_path',
                      default='output',
                      help='path to save results',
                      metavar='OUTPUT_PATH')    
    
    options, args = parser.parse_args()
    
    input_filename_x = 'train_data.csv'
    input_filename_y = 'train_labels.csv'    
    test_input_filename = 'test_data.csv'
    
    model_filename = 'model.yaml'
    weights_filename = 'weights.h5'
    
    io = lib.io.IO()
    viz = lib.viz.Viz()
    cl = lib.cl.CL(io, viz)
    
    # Read data
    print "Reading train data..."
    X, y = io.read_data(input_filename_x, input_filename_y)
    y = io.shift_v(y, shift=-1)    
    
    print "Reading test data..."
    test_x = io.read_data(test_input_filename, None)
    
    print "There are " + str(len(X)) + " samples in the train set."
    print "There are " + str(len(test_x)) + " samples in the test set."
    
    test_x = np.matrix(test_x)
    test_ids = range(1, len(test_x)+1)    

    # load from file
    if options.load_path is not None:
        model = load_model(options.load_path+'/'+model_filename, options.load_path+'/'+weights_filename)
    # train
    else:
        # Remove outliers
        X,y = cl.lof(np.matrix(X), np.matrix(y))

        # Suffle
        X, y = io.shuffle(X, y)
        
        # Pick val and train set
        val_ids, val_x, val_y = io.pick_set(X, y, 526)
        train_ids, train_x, train_y = io.pick_set(X, y, 3400)    

        # train
        model = train_model(train_x, train_y, val_x, val_y)
        
        # Save model
        #full_model_filename = options.model_path + '/'+model_filename
        #full_weights_filename = options.model_path + '/'+weights_filename
        #with open(full_model_filename, 'w') as f:
         #   f.write(model.to_yaml())
         #   model.save_weights(full_weights_filename)

        # Print metrics
        #viz.plot_nn_perf(history, options.output_path+'/nn_perf.png')

    # visualize model    
    viz.plot_model(model, options.output_path+'/model.png')
    # viz.plot_all_feature_maps(model, test_x[:12], options.output_path+'/')
    
    # predict
    pred_proba = predict(model, test_x)
    pred_class = np.argmax(pred_proba, axis=1)    
    pred_class = io.shift_v(pred_class, shift=1)    

    # Output
    io.write_classes(options.output_path+'/classes_result.csv', test_ids, pred_class)
    io.write_probabilities(options.output_path+'/probabilities_result.csv', test_ids, pred_proba)



