# -*- coding: utf-8 -*-
import sys, copy
import numpy as np
import lib.io
import lib.viz
import lib.nn
import lib.cl

def main(argv):

    input_filename_x = 'train_data.csv'
    input_filename_y = 'train_labels.csv'    

    test_input_filename = 'test_data.csv'

    svc_model_filename = 'svc_classif.pkl'
    lr_model_filename = 'lr_classif.pkl'
    rfc_model_filename = 'rfc_classif.pkl'
    rfc_feat_imp_filename = 'rfc_feat_imp.png'
    model_comp_result_chart_filename = 'method_comp_res.png'
    
    nn_model_filename = 'nn1.pkl'
    
    io = lib.io.IO()
    viz = lib.viz.Viz()    
    nn = lib.nn.NN(io, viz)
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

    # PCA etc.
    X = cl.pca(np.matrix(X), 'pca_explained_variance.png').tolist()
    test_x = cl.pca(test_x, None).tolist()
    
    # Split data to train and validation set
    # mini_batches
    #ids, batches_x, batches_y = io.split_data(X, y, 100, 100)
    
    val_ids, val_x, val_y = io.pick_set(X, y, 563)
    train_ids, train_x, train_y = io.pick_set(X, y, 3800)

    nn.initialize(train_x.shape[1], nn1=18, nn2=9, alpha=0.01) #, filename=nn_model_filename)
       
    # Train
    pred, proba, acc = nn.predict(train_x, train_y)
    print("Train set classification accuray before training: %.4f"%acc)

    nn.train(train_x, train_y, val_x, val_y, training_steps=100000)
    nn.save_nn(nn_model_filename)
        
    # validate
    pred, proba, acc = nn.predict(train_x, train_y)
    print("Train set classification accuray after training: %.4f"%acc)
    pred, proba, acc = nn.predict(val_x, val_y)
    print("Validation set classification accuray after training: %.4f"%acc)        
    
    # Draw some results
    # viz.model_comp_results(results, model_comp_result_chart_filename)    
    
    # predict
    pred_class, pred_proba, _ = nn.predict(test_x)
    pred_class = io.shift_v(pred_class, shift=1)
    
    # Output
    io.write_classes('nn_classes_sub_result.csv', test_ids, pred_class)
    io.write_probabilities('nn_probabilities_sub_result.csv', test_ids, pred_proba)

if __name__ == "__main__":
    main(sys.argv[1:])
