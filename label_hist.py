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
    
    io = lib.io.IO()
    viz = lib.viz.Viz()
    cl = lib.cl.CL(io, viz)
    
    # Read data
    X, y = io.read_data(input_filename_x, input_filename_y)
    test_x = io.read_data(test_input_filename, None)
    
    print "There are " + str(len(X)) + " samples in the train set."
    print "There are " + str(len(test_x)) + " samples in the test set."
    
    viz.label_hist(y, 'label_hist.png')
        
if __name__ == "__main__":
    main(sys.argv[1:])

