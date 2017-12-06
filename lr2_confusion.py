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

    lr_model_filename = 'lr2_classif.pkl'
    model_comp_result_chart_filename = 'method_comp_res.png'
    
    io = lib.io.IO()
    viz = lib.viz.Viz()
    cl = lib.cl.CL(io, viz)
    
    # Read data
    X, y = io.read_data(input_filename_x, input_filename_y)
    test_x = io.read_data(test_input_filename, None)
    X_ = copy.deepcopy(X)
    y_ = copy.deepcopy(y)
    
    print "There are " + str(len(X)) + " samples in the train set."
    print "There are " + str(len(test_x)) + " samples in the test set."
    
    test_x = np.matrix(test_x)
    test_ids = range(1, len(test_x)+1)    

    cl.lr_cl_load(lr_model_filename)    
    
    # predict
    pred_class, pred_proba = cl.lr_cl_pred(X)

    viz.plot_confusion_matrix(y, pred_class, np.arange(1,11))
    viz.plot_confusion_matrix(y, pred_class, np.arange(1,11), normalize=True, filename='confusion_matrix_norm.png')
    
if __name__ == "__main__":
    main(sys.argv[1:])

