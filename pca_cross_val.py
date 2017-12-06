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

    svc_model_filename = 'svc_classif.pkl'
    lr_model_filename = 'lr_classif.pkl'
    rfc_model_filename = 'rfc_classif.pkl'
    rfc_feat_imp_filename = 'rfc_feat_imp.png'
    model_comp_result_chart_filename = 'method_comp_res.png'
    
    io = lib.io.IO()
    viz = lib.viz.Viz()
    
    # Read data
    X_o, y_o = io.read_data(input_filename_x, input_filename_y)
    test_x = io.read_data(test_input_filename, None)

    print "There are " + str(len(X_o)) + " samples in the train set."
    print "There are " + str(len(test_x)) + " samples in the test set."

    SVC_ll , SVC_a, RFC_ll, RFC_a, LR_ll, LR_a = [], [], [], [], [], []

    comps = [10, 20, 50, 100, 150, 200, 264]
    for s in comps:
        print("Amount of components: %d"%s)
        cl = lib.cl.CL(io, viz)
        X = copy.deepcopy(X_o)
        y = copy.deepcopy(y_o)
        
        test_x = np.matrix(test_x)
        test_ids = range(1, len(test_x)+1)    
        
        # Remove outliers
        X, y = cl.lof(np.matrix(X), np.matrix(y))
        
        # Shuffle
        X, y = io.shuffle(X, y)

        # PCA
        X = cl.pca(np.matrix(X), components=s, filename=None).tolist()
        # test_x = cl.pca(np.matrix(test_x), components=s, filename=None).tolist()

        val_ids, val_x, val_y = io.pick_set(X, y, 726)
        train_ids, train_x, train_y = io.pick_set(X, y, 3200)
    
        # Train
        cl.lr_cl_train(train_x, train_y, filename=lr_model_filename)

        # Validate
        ll, a = cl.lr_cl_val(val_x, val_y)
        LR_ll.append(ll)
        LR_a.append(a)

    # Draw some results
    viz.cross_results(comps, LR_ll, LR_a, 'pca_cross_val.png')
        
if __name__ == "__main__":
    main(sys.argv[1:])

