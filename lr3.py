# -*- coding: utf-8 -*-
import sys, copy
import numpy as np
import lib.io
import lib.viz
import lib.cl

def main(argv):

    input_filename_x = 'train_data.csv'
    input_filename_y = 'train_labels.csv'    
    test_input_filename = 'test_data.csv'

    lr_model_filename = 'lr3_classif.pkl'
    model_comp_result_chart_filename = 'method_comp_res.png'
    
    io = lib.io.IO()
    viz = lib.viz.Viz()
    cl = lib.cl.CL(io, viz)
    
    # Read data
    X, y = io.read_data(input_filename_x, input_filename_y)
    test_x = io.read_data(test_input_filename, None)
    
    print "There are " + str(len(X)) + " samples in the train set."
    print "There are " + str(len(test_x)) + " samples in the test set."
    
    test_x = np.matrix(test_x)
    test_ids = range(1, len(test_x)+1)    

    # Remove outliers
    X, y = cl.lof(np.matrix(X), np.matrix(y))

    # Shuffle
    X, y = io.shuffle(X, y)

    # PCA
    X = cl.pca(np.matrix(X), components=150).tolist()
    test_x = cl.pca(np.matrix(test_x), components=150).tolist()

    # Split data to train and validation set
    val_ids, val_x, val_y = io.pick_set(X, y, 726)
    train_ids, train_x, train_y = io.pick_set(X, y, 3200)
    
    # Train
    cl.lr_cl_train(train_x, train_y, filename=lr_model_filename)
    # cl.lr_cl_load(lr_model_filename)    

    # Validate
    cl.lr_cl_val(val_x, val_y)
    
    # predict
    pred_class, pred_proba = cl.lr_cl_pred(test_x)
    
    # Output
    io.write_classes('classes_lr3_result.csv', test_ids, pred_class)
    io.write_probabilities('probabilities_lr3_result.csv', test_ids, pred_proba)

if __name__ == "__main__":
    main(sys.argv[1:])

