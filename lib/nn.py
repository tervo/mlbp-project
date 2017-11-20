# -*- coding: utf-8 -*-
import sys
from theano import tensor as T
import numpy as np
import theano
import cPickle
import matplotlib.pyplot as plt
import matplotlib as mt
import matplotlib.gridspec as gridspec

class NN:

    n_classes = 10
    
    def __init__(self, io, viz):
        self.io = io
        self.viz = viz

    def initialize(self, xdim, nn1=10, nn2=5, alpha=0.2, filename=None):

        print "Initializing NN..."
        
        # define input and output variables in theano
        self.alpha = alpha
        xx = T.matrix('x')
        yy = T.ivector('y')        
            
        # layer 01 parameter declaration & initialization (weights/bias)
        np.random.seed(1232)
        self.w_1 = theano.shared(np.random.randn(xdim, nn1), name='w_1') 
        self.b_1 = theano.shared(np.zeros((nn1,)), name='b_1') 

        # layer 02 parameter declaration & initialization (weights/bias) 
        np.random.seed(1232)    
        self.w_2 = theano.shared(np.random.randn(nn1, nn2), name='w_2')
        self.b_2 = theano.shared(np.zeros((nn2,)), name='b_2')

        # output layer parameter declaration & initialization (weights/bias) 
        np.random.seed(1232)
        self.w_out = theano.shared(np.random.randn(nn2, self.n_classes), name='w_out') 
        self.b_out = theano.shared(np.zeros(self.n_classes), name='b_out') 

        # hidden layer output
        self.h_out_1 = theano.tensor.nnet.sigmoid(T.dot(xx, self.w_1) + self.b_1) 
        self.h_out_2 = theano.tensor.nnet.sigmoid(T.dot(self.h_out_1, self.w_2) + self.b_2) 
        
        # perceptron predictions
        p = T.nnet.softmax(T.dot(self.h_out_2, self.w_out) + self.b_out) 
        y_pred = T.argmax(p, axis=1)
                
        # cross-entropy as cost function
        #cost = T.mean(T.nnet.binary_crossentropy(y_pred, y)) 
        # cost = T.mean((y_pred - y)**2)
        cost = -T.mean(T.log(p)[T.arange(p.shape[0]), yy])

        # gradient computation
        gw_1, gb_1, gw_2, gb_2, gw_out, gb_out = T.grad(cost, [self.w_1, self.b_1, self.w_2, self.b_2, self.w_out, self.b_out])
        
        # train_model theano function
        # Note : outputs should return following in order
        #      : [prediction vector, error/cost scalar,
        #        1st hidden layer activation vector, 2nd hidden layer activation vector]
        self.train_model = theano.function(
            inputs  = [xx,yy],
            outputs = [y_pred, cost, self.h_out_1, self.h_out_2],
            updates = [(self.w_1, self.w_1 - self.alpha * gw_1), 
                       (self.b_1, self.b_1 - self.alpha * gb_1),
                       (self.w_2, self.w_2 - self.alpha * gw_2),
                       (self.b_2, self.b_2 - self.alpha * gb_2),
                       (self.w_out, self.w_out - self.alpha * gw_out), 
                       (self.b_out, self.b_out - self.alpha * gb_out)]
        )

        # cost function
        self.cost_function = theano.function(inputs=[xx,yy], outputs=cost)
        
        # predict model functions
        self.predict_proba = theano.function(inputs=[xx], outputs=[p])
        self.predict_model = theano.function(inputs=[xx], outputs=[y_pred])

        if filename is not None:
            self.load_nn(filename)

    def get_cost(self, X, Y):
        Y = np.asarray(Y, dtype=np.int32).flatten()        
        return self.cost_function(X, Y)
        
    def predict(self, X, Y=None):
        pred = (np.round(self.predict_model(X)).reshape((1,-1)))
        proba = self.predict_proba(X)[0]
        
        acc = None
        if Y is not None:
            acc = np.mean(pred == Y)
                    
        return pred.tolist()[0], proba, acc    

    def train(self, Xtrain, Ytrain, Xtest, Ytest, training_steps=50000, plot_prefix=''):

        print "Training with " + str(training_steps) + " steps"

        Ytrain = np.asarray(Ytrain, dtype=np.int32).flatten()
        Ytest = np.asarray(Ytest, dtype=np.int32).flatten()
        cost_train_vec = np.array([])
        cost_test_vec = np.array([])

        for i in np.arange(training_steps):
        
            pred_train, cost_train, nactivation1, nactivation2 = self.train_model(
                Xtrain, Ytrain)
            cost_train_vec = np.append(cost_train_vec, cost_train)
            
            # get predictions, cost on test set
            pred_test = self.predict_model(Xtest)
            cost_test = self.cost_function(Xtest,Ytest)
            cost_test_vec = np.append(cost_test_vec, cost_test)
        
            # printing
            total = 10000
            if training_steps < total:
                total = training_steps
            if i % 10000 == 0:
                print("Iteration %6s -- "%i,'Training cost: ',"%4.4f"%cost_train)

        print("final train set cost : %.4f"%cost_train)
        print("final test set cost  : %.4f"%cost_test)

        self.viz.plot_learning(cost_train_vec, cost_test_vec, plot_prefix+'nn1-learning.png')
        self.viz.plot_activation(nactivation1, nactivation2, plot_prefix+'nn1-activation.png')
        
    def load_nn(self, filename):
        save_file = open(filename)
        self.w_1.set_value(cPickle.load(save_file), borrow=True)
        self.b_1.set_value(cPickle.load(save_file), borrow=True)
        self.w_2.set_value(cPickle.load(save_file), borrow=True)
        self.b_2.set_value(cPickle.load(save_file), borrow=True)
        self.w_out.set_value(cPickle.load(save_file), borrow=True)
        self.b_out.set_value(cPickle.load(save_file), borrow=True)

        
    def save_nn(self, filename):        
        save_file = open(filename, 'wb')
        cPickle.dump(self.w_1.get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
        cPickle.dump(self.b_1.get_value(borrow=True), save_file, -1)  
        cPickle.dump(self.w_2.get_value(borrow=True), save_file, -1)  
        cPickle.dump(self.b_2.get_value(borrow=True), save_file, -1)
        cPickle.dump(self.w_out.get_value(borrow=True), save_file, -1)
        cPickle.dump(self.b_out.get_value(borrow=True), save_file, -1)  
        save_file.close()

        
    def print_progress(self, iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
        """
        Call in a loop to create terminal progress bar
        @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
        """
        str_format = "{0:." + str(decimals) + "f}"
        percents = str_format.format(100 * (iteration / float(total)))
        filled_length = int(round(bar_length * iteration / float(total)))
        bar = '=' * filled_length + '-' * (bar_length - filled_length)

        sys.stdout.write('\r')
        sys.stdout.write("%s |%s| %s%s %s" % (prefix, bar, percents, '%', suffix))
        
        if iteration == total:
            sys.stdout.write('\n')
            
        sys.stdout.flush()
    def classificationTwoHiddenLayerNN(self,
                                       Xtrain,
                                       Ytrain,
                                       Xtest,
                                       Ytest,
                                       nn1=4,
                                       nn2=3,
                                       training_steps=50000,
                                       alpha=0.2,
                                       plot_filename=None):
        '''
        Input:
        Xtrain   : N x D    : traning set features
        Ytrian   : N x 1    : training set target
        Xtest    : M x D    : test set feaures
        Ytest    : M x 1    : test set target
        nn1      : scalar   : no. of neurons to be used in first hidden layer
        nn2      : scalar   : no. of neurons to be used in second hidden layer
        training_steps : scalar : no. of training iteration steps
        alpha    : scalar   : learning rate    
        plot_filename : string   : filename for the plot
        '''
        print("*** running ***")

        print("Ytrain shape: ",Ytrain.shape)
        print("Ytest shape: ",Ytest.shape)

        # define input and output variables in theano
        n_classes = 10
        x = T.matrix('x')
        y = T.ivector('y')        

        xdim = Xtrain.shape[1]  # number of features in the data
        Ytrain = np.asarray(Ytrain, dtype=np.int32).flatten()
        Ytest = np.asarray(Ytest, dtype=np.int32).flatten()
        
        print("Ytrain shape: ",Ytrain.shape)
        print("Ytest shape: ",Ytest.shape)
        
        # Ytrain.reshape(Xtrain.shape[0],0)
                
        # layer 01 parameter declaration & initialization (weights/bias)
        np.random.seed(1232)
        self.w_1 = theano.shared(np.random.randn(xdim, nn1), name='w_1') 
        self.b_1 = theano.shared(np.zeros((nn1,)), name='b_1') 

        # layer 02 parameter declaration & initialization (weights/bias) 
        np.random.seed(1232)    
        self.w_2 = theano.shared(np.random.randn(nn1, nn2), name='w_2')
        self.b_2 = theano.shared(np.zeros((nn2,)), name='b_2')

        # output layer parameter declaration & initialization (weights/bias) 
        np.random.seed(1232)
        self.w_out = theano.shared(np.random.randn(nn2,n_classes), name='w_out') 
        self.b_out = theano.shared(np.zeros(n_classes), name='b_out') 

        # hidden layer output
        self.h_out_1 = theano.tensor.nnet.sigmoid(T.dot(x, self.w_1) + self.b_1)
        self.h_out_1 = theano.tensor.nnet.sigmoid(T.dot(self.h_out_1, self.w_2) + self.b_2)
        #self.h_out_1 = T.tanh(T.dot(x, self.w_1) + self.b_1) 
        #self.h_out_2 = T.tanh(T.dot(self.h_out_1, self.w_2) + self.b_2) 
        
        # perceptron predictions
        p = T.nnet.softmax(T.dot(self.h_out_2, self.w_out) + self.b_out) 
        y_pred = T.argmax(p, axis=1)
                
        # cross-entropy as cost function
        #cost = T.mean(T.nnet.binary_crossentropy(y_pred, y)) 
        # cost = T.mean((y_pred - y)**2)
        cost = T.mean(T.log(p)[T.arange(p.shape[0]), y])

        # gradient computation
        gw_1, gb_1, gw_2, gb_2, gw_out, gb_out = T.grad(cost, [self.w_1, self.b_1, self.w_2, self.b_2, self.w_out, self.b_out])
        
        # train_model theano function
        # Note : outputs should return following in order
        #      : [prediction vector, error/cost scalar,
        #        1st hidden layer activation vector, 2nd hidden layer activation vector]
        train_model = theano.function(
            inputs  = [x,y],
            outputs = [y_pred, cost, self.h_out_1, self.h_out_2],
            updates = [(self.w_1, self.w_1 - alpha * gw_1), 
                       (self.b_1, self.b_1 - alpha * gb_1),
                       (self.w_2, self.w_2 - alpha * gw_2),
                       (self.b_2, self.b_2 - alpha * gb_2),
                       (self.w_out, self.w_out - alpha * gw_out), 
                       (self.b_out, self.b_out - alpha * gb_out)]
        )

        # cost function
        cost_function = theano.function(inputs=[x,y], outputs=cost)
        
        # predict model functions
        predict_proba = theano.function(inputs=[x], outputs=[p])
        predict_model = theano.function(inputs=[x], outputs=[y_pred])        
        
        # accumulate error over iterations on traning and test set in a vector
        cost_train_vec = np.array([])
        cost_test_vec = np.array([])

        train_predictions = (np.round(predict_model(Xtrain)).reshape((1,-1)))
        train_accuracy = np.mean(train_predictions == Ytrain)
        print("Train set classification accuracy before training: %.4f"%train_accuracy)        
        
        # training iterations begin
        for i in np.arange(training_steps):
        
            pred_train, cost_train, nactivation1, nactivation2 = train_model(
                Xtrain, Ytrain)
            cost_train_vec = np.append(cost_train_vec, cost_train)
            
            # get predictions, cost on test set
            pred_test = predict_model(Xtest)
            cost_test = cost_function(Xtest,Ytest)
            cost_test_vec = np.append(cost_test_vec, cost_test)
        
            # printing
            total = 10000
            if training_steps < total:
                total = training_steps
            if i % 10000 == 0:
                print("Iteration %6s -- "%i,'Training cost: ',"%4.4f"%cost_train)

        print("final train set cost : %.4f"%cost_train)
        print("final test set cost  : %.4f"%cost_test)

        self.viz.plot_learning(cost_train_vec, cost_test_vec, 'nn1-learning.png')
        self.viz.plot_activation(nactivation1, nactivation2, 'nn1-activation.png')
        
        # compute classification accuracies
        train_predictions = (np.round(predict_model(Xtrain)).reshape((1,-1)))
        train_accuracy = np.mean(train_predictions == Ytrain)
        print("final train set classification accuracy : %.4f"%train_accuracy)
        
        test_predictions = np.round(pred_test).reshape((1,-1))
        test_accuracy = np.mean(test_predictions == Ytest)
        print("Final test set classification accuracy : %.4f"%test_accuracy)
