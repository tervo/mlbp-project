# -*- coding: utf-8 -*-
import sys, copy
import numpy as np
import lib.io
import lib.viz
import lib.nn

def main(argv):

    input_filename_x = 'train_data.csv'
    input_filename_y = 'train_labels.csv'    

    test_input_filename = 'test_data.csv'

    svc_model_filename = 'svc_classif.pkl'
    lr_model_filename = 'lr_classif.pkl'
    rfc_model_filename = 'rfc_classif.pkl'
    rfc_feat_imp_filename = 'rfc_feat_imp.png'
    model_comp_result_chart_filename = 'method_comp_res.png'
    
    nn_model_filename = 'kf-nn1.pkl'
    
    io = lib.io.IO()
    viz = lib.viz.Viz()    
    nn = lib.nn.NN(io, viz)
    
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
    
    # Split data to train and validation set
    # mini_batches
    num_of_batches = 5
    ids, batches_x, batches_y = io.split_data(X, y, num_of_batches)
    
    #val_ids, val_x, val_y = io.pick_set(X, y, 563)
    #train_ids, train_x, train_y = io.pick_set(X, y, 3800)
       
    # Train
    training_errors = []
    validation_errors = []
    model_sizes = [3, 9, 27, 81, 243]
    
    for nn2 in model_sizes:
        nn1 = nn2*2
        
        avg_error_validation = 0
        avg_error_train = 0
        for batch_num in range(num_of_batches):
            nn.initialize(batches_x[0].shape[1], nn1=nn1, nn2=nn2)
            val_x = batches_x[batch_num]
            val_y = batches_y[batch_num]            

            # train
            for train_batch_num in range(num_of_batches):
                if train_batch_num == batch_num: continue                                
                nn.train(batches_x[train_batch_num], batches_y[train_batch_num], val_x, val_y, training_steps=1000, plot_prefix='k-fold-')

            # Calculate average training error with optimal w
            train_error = 0
            for train_batch_num in range(num_of_batches):
                if train_batch_num == batch_num: continue                    
                c = nn.get_cost(batches_x[train_batch_num], batches_y[train_batch_num])
                train_error += (c - train_error)/(train_batch_num + 1)                    

            avg_error_train += (c - avg_error_train)/(batch_num + 1)

            # Validate
            error_validation = nn.get_cost(val_x, val_y)
            avg_error_validation += (error_validation - avg_error_validation)/(batch_num+1)

            # Output
            print 'Batch ' + str(batch_num)+' validation error: '+str(error_validation)
            print 'AVG validation error after validation batch: ' + str(batch_num)+': '+str(avg_error_validation)
            print ' '
            print '-----'

        validation_errors.append(avg_error_validation)
        training_errors.append(avg_error_train)
        
    nn.save_nn(nn_model_filename)
            
    # Draw some results
    # viz.model_comp_results(results, model_comp_result_chart_filename)    
    viz.model_size_comp(model_sizes, validation_errors, training_errors, 'nn1_model_size_comp.png')

if __name__ == "__main__":
    main(sys.argv[1:])
