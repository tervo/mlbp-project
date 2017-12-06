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
    
    for s in range(50):
        print("Iteration %d"%s)
        cl = lib.cl.CL(io, viz)
        X = copy.deepcopy(X_o)
        y = copy.deepcopy(y_o)
        X_ = copy.deepcopy(X_o)
        y_ = copy.deepcopy(y_o)
    
        
        test_x = np.matrix(test_x)
        test_ids = range(1, len(test_x)+1)    
        
        # Remove outliers
        X, y = cl.lof(np.matrix(X), np.matrix(y))
        X_, y_ = cl.lof(np.matrix(X_), np.matrix(y_))
        
        # Shuffle
        X, y = io.shuffle(X, y)
        X_, y_ = io.shuffle(X_, y_)

        # PCA
        X = cl.pca(np.matrix(X), 'pca_explained_variance.png').tolist()
        test_x = cl.pca(np.matrix(test_x), None).tolist()

        val_ids, val_x, val_y = io.pick_set(X, y, 726)
        _, no_pca_val_x, no_pca_val_y = io.pick_set(X_, y_, 726)
        train_ids, train_x, train_y = io.pick_set(X, y, 3200)
        _, no_pca_train_x, no_pca_train_y = io.pick_set(X_, y_, 3200)
    
        # Train
        cl.svc_cl_train(train_x, train_y, filename=svc_model_filename)
        cl.lr_cl_train(train_x, train_y, filename=lr_model_filename)
        cl.rfc_cl_train(no_pca_train_x, no_pca_train_y,
                        filename=rfc_model_filename,
                        feat_imp_plot_filename=rfc_feat_imp_filename)
        
        # validate
        ll, a = cl.svc_cl_val(val_x, val_y)
        SVC_ll.append(ll)
        SVC_a.append(a)

        ll, a = cl.lr_cl_val(val_x, val_y)
        LR_ll.append(ll)
        LR_a.append(a)

        ll, a = cl.rfc_cl_val(no_pca_val_x, no_pca_val_y)
        RFC_ll.append(ll)
        RFC_a.append(a)
            
    # Draw some results
    results = {'SVC': (sum(SVC_ll)/len(SVC_ll), sum(SVC_a)/len(SVC_a)),
               'Logistic Regression': (sum(LR_ll)/len(LR_ll), sum(LR_a)/len(LR_a)),
               'Random Forest Classifier': (sum(RFC_ll)/len(RFC_ll), sum(RFC_a)/len(RFC_a)) }
    viz.model_comp_results(results, model_comp_result_chart_filename)    
        
    # predict
    pred_class, pred_proba = cl.lr_cl_pred(test_x)
    
    # Output
    io.write_classes('classes_sub_result.csv', test_ids, pred_class)
    io.write_probabilities('probabilities_sub_result.csv', test_ids, pred_proba)

if __name__ == "__main__":
    main(sys.argv[1:])

