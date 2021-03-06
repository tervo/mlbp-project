# -*- coding: utf-8 -*-
import sys, re
import numpy as np
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix

class CL:
    
    class_count = 10

    def __init__(self, io, viz):
        self.io = io
        self.viz = viz
        self.svc_classif = OneVsRestClassifier(SVC(kernel='linear', probability=True), n_jobs=-1)
        self.lr_classif = OneVsRestClassifier(LogisticRegression(tol=0.00001, max_iter=10000, penalty='l2', C=1.), n_jobs=-1)
        self.rfc_classif = RandomForestClassifier(n_jobs=-1)

    ################################################################################    

    def pca(self, X, components=20, filename=None):
        print "Doing PCA analysis..."

        # For visualizing purposes
        if filename is not None:
            pca = decomposition.PCA(whiten=True)
            pca.fit(X)
            self.viz.explained_variance(pca, filename)

        # With actual number of components
        #pca = decomposition.PCA(whiten=True, n_components=264)
        pca = decomposition.PCA(n_components=components)
        X = pca.fit_transform(X)
        print("X shape after PCA detection %d, %d"%X.shape)
        return X

    def lof(self, X, y):
        print "Doing LOF outlier detection..."

        r,f = X.shape
        clf = LocalOutlierFactor(n_neighbors=20, n_jobs=-1)
        y_pred = clf.fit_predict(X)
        X = X[y_pred>0,:]
        y = y[:, y_pred>0]
        
        print "Removed "+ str(r - X.shape[0]) + " outliers"
        print("X shape after outlier detection %d, %d"%X.shape)
        print("y shape after outlier detection %d, %d"%y.shape)
        return X.tolist(), y.tolist()[0]
    
    def confusion(self, y_true, y_pred):
        print "Creating confusion matrix..."

        conf = confusion_matrix(y_true, y_pred)

        i = 0
        for true_value in conf:
            i += 1
            print "& \\textbf{"+str(i)+"} & " + " & ".join(map(str, true_value.tolist())) + "\\\\"

        conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
        norm = []
        for row in conf:
            nrow = []
            [nrow.append(round(float(i), 2)) for i in row]
            norm.append(nrow)

        print "Normalized: "
        i = 0
        for true_value in norm:
            i += 1
            print "& \\textbf{"+str(i)+"} & " + " & ".join(map(str, true_value)) + "\\\\"
        
        
    # SVC
    ################################################################################
    
    def svc_cl_load(self, filename):
        self.svc_classif = self.io.load_model(filename)
        
    def svc_cl_train(self, x, y, filename=None):
        """ Train simple linear SVC 
        x:  [matrix, (n,k)]  n samples with k features
        y:  [matrix, (n,1)]  n labels
        """
        
        print "Training SVC with " + str(len(x)) + " samples..."        
        self.svc_classif.fit(x, y)
        if filename is not None:
            self.io.save_model(self.svc_classif, filename)
        
        print "Training SVC complete"        
        
    def svc_cl_pred(self, x):
        """ Predict with the trained classifier 
        x:  [matrix, (n,k)]  n samples with k features
        return [pred_class, pred_proba]
        """
        
        # Validate
        pred_proba = self.svc_classif.predict_proba(x)
        probabilities = np.zeros((len(x), self.class_count), float)

        rownum = 0
        for row in pred_proba:            
            c = 0
            for i in row:
                col = int(self.svc_classif.classes_[c]-1)
                probabilities[rownum, col] = i
                c += 1
            rownum += 1
        
        # print probabilities
        # print "Found following classes:"
        # print self.svc_classif.classes_
        
        pred_class = self.svc_classif.predict(x)
        
        return pred_class, probabilities

    def svc_cl_val(self, x, y):        
        """ Validate training
        x:  [matrix, (n,k)]  n samples with k features
        y:  [matrix, (n,1)]  n labels
        return (float) score
        """
        
        score = self.svc_classif.score(x, y)
        pred_proba = self.svc_classif.predict_proba(x)
        loss = log_loss(y, pred_proba)            

        print "Validation results for SVC"
        print "Validation score: " + str(score)
        print "Validation loss: "+ str(loss)
        
        return loss, score

    # Linear Regression
    ################################################################################
    def lr_cl_load(self, filename):
        self.lr_classif = self.io.load_model(filename)
    
    def lr_cl_train(self, x, y, filename=None):
        """ Train simple logistic regression 
        x:  [matrix, (n,k)]  n samples with k features
        y:  [matrix, (n,1)]  n labels
        """

        print "Training LogisticRegression with " + str(len(x)) + " samples..."        
        self.lr_classif.fit(x, y)
        if filename is not None:
            self.io.save_model(self.lr_classif, filename)
        
        print "Training LogisticRegression complete"        

            
    def lr_cl_pred(self, x):
        """ Predict with the trained classifier 
        x:  [matrix, (n,k)]  n samples with k features
        return [pred_class, pred_proba]
        """
        
        pred_proba = self.lr_classif.predict_proba(x)
        probabilities = np.zeros((len(x), self.class_count), float)

        rownum = 0
        for row in pred_proba:            
            c = 0
            for i in row:
                col = int(self.lr_classif.classes_[c]-1)
                probabilities[rownum, col] = i
                c += 1
            rownum += 1
        
        # print probabilities
        # print "Found following classes:"
        # print self.lr_classif.classes_
        
        pred_class = self.lr_classif.predict(x)
        
        return pred_class, probabilities

    def lr_cl_val(self, x, y):        
        """ Validate training
        x:  [matrix, (n,k)]  n samples with k features
        y:  [matrix, (n,1)]  n labels
        return (float) score
        """
        
        score = self.lr_classif.score(x, y)
        pred_proba = self.lr_classif.predict_proba(x)
        loss = log_loss(y, pred_proba)        

        print "Validation results for linear regression"        
        print "Validation score: " + str(score)
        print "Validation loss: "+ str(loss)
        
        return loss, score


    # RandomForestClassfier
    ################################################################################
    def rfc_cl_load(self, filename):
        self.rfc_classif = self.io.load_model(filename)

    def rfc_cl_train(self, x, y, filename=None, feat_imp_plot_filename=None):
        """ Train simple logistic regression 
        x:  [matrix, (n,k)]  n samples with k features
        y:  [matrix, (n,1)]  n labels
        """
        
        print "Training RandomForestClassifier with " + str(len(x)) + " samples..."
        self.rfc_classif.fit(x, np.asarray(y).ravel())
        if filename is not None:
            self.io.save_model(self.rfc_classif, filename)

        if feat_imp_plot_filename is not None:
            self.viz.rfc_feature_importance(self.rfc_classif.feature_importances_, feat_imp_plot_filename)
            
            
        print "Training RandomForestClassfier complete"        
            
    def rfc_cl_pred(self, x, feat_imp_plot_filename = None):
        """ Predict with the trained classifier 
        x:  [matrix, (n,k)]  n samples with k features
        return [pred_class, pred_proba]
        """
        
        pred_proba = self.rfc_classif.predict_proba(x)
        probabilities = np.zeros((len(x), self.class_count), float)

        rownum = 0
        for row in pred_proba:            
            c = 0
            for i in row:
                col = int(self.rfc_classif.classes_[c]-1)
                probabilities[rownum, col] = i
                c += 1
            rownum += 1
        
        # print probabilities
        # print "Found following classes:"
        # print self.rfc_classif.classes_
                
        pred_class = self.rfc_classif.predict(x)
        
        return pred_class, probabilities

    def rfc_cl_val(self, x, y):        
        """ Validate training
        x:  [matrix, (n,k)]  n samples with k features
        y:  [matrix, (n,1)]  n labels
        return (float) score
        """
        
        score = self.rfc_classif.score(x, y)
        pred_proba = self.rfc_classif.predict_proba(x)
        loss = log_loss(y, pred_proba)        

        print "Validation results for RFC"
        print "Validation score: " + str(score)
        print "Validation loss: "+ str(loss)
        
        return loss, score

    
