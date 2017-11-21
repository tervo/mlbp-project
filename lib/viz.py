# -*- coding: utf-8 -*-
import sys, re
import numpy as np
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from keras.utils import plot_model

class Viz:

    def scree_plot(self, X, filename):
        rows, cols = X.shape
        U, S, V = np.linalg.svd(X) 
        eigvals = S**2 / np.cumsum(S)[-1]

        fig = plt.figure(figsize=(8,5))
        sing_vals = np.arange(cols) + 1
        plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')

        leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, 
                         shadow=False, prop=mlp.font_manager.FontProperties(size='small'),
                         markerscale=0.4)
        leg.get_frame().set_alpha(0.4)
        leg.draggable(state=True)
        
        plt.savefig(filename)

    def model_comp_results(self, results, filename):
        fig, ax1 = plt.subplots(figsize=(12,8))

        x, y1, y2 = [], [], []
        
        for model,values in results.items():
            x.append(model)
            y1.append(list(values)[0])
            y2.append(list(values)[1])

        ind = np.arange(len(x))
        width = 0.35
            
        rects1 = ax1.bar(ind, y1, width, color='y')

        ax2 = ax1.twinx()
        rects2 = ax2.bar(ind + width+0.05, y2, width, color='r')


        def autolabel(rects, ax):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., 1.02 * height,
                        '%.2f' % height,
                        ha='center', va='bottom')
                
        autolabel(rects1, ax1)
        autolabel(rects2, ax2)        
        
        ax1.set_xticks(ind + width/2)
        ax1.set_xticklabels(x)

        ax1.set_ylabel('Prediction score')
        ax2.set_ylabel('Logistic loss for prediction probabilities')
        
        plt.savefig(filename)

        print "Saved method comparison chart to " + filename
        
        
            
    def rfc_feature_importance(self, data, filename):
        fig, ax1 = plt.subplots(figsize=(12,8))

        plt.clf()
        plt.bar(range(0,len(data)), data)
        plt.xlabel('components')
        plt.ylabel('importance')
        
        plt.savefig(filename)

        print "Saved feature importance to "+filename
        

    def explained_variance(self, pca, filename):
        
        fig, ax1 = plt.subplots(figsize=(16,10))

        plt.clf()
        plt.xticks(np.arange(0, pca.n_components_, 10))
        plt.grid()        

        plt.plot(pca.explained_variance_, linewidth=1)
        
        #ax2 = ax1.twinx()        
        #ax2.plot(pca.explained_variance_ratio_, linewidth=1)

        plt.axis('tight')
        plt.xlabel('n components')
        plt.ylabel('explained variance')
        #ax2.set_ylabel('explained variance ratio')
        
        plt.savefig(filename)

        print "Saved explained variance to "+filename
    
    def scatter_classes(self, X, Y, filename):

        plt.figure()

        
        
        plt.savefig(filename)


    def plot_learning(self, cost_train_vec, cost_test_vec, filename):
        
        fig, ax1 = plt.subplots(figsize=(16,10))

        plt.clf()
        plt.grid()        
        
        #ax1 = fig.add_subplot(gs00[1, 0])  #, adjustable='box-forced'
        plt.plot(
            np.arange(cost_train_vec.shape[0]),
            cost_train_vec,
            c="#27ae61",
            label="train")
        plt.plot(
            np.arange(cost_test_vec.shape[0]),
            cost_test_vec,
            c="#c1392b",
            label="test")
        plt.xlabel("iterations")
        plt.ylabel("cost function")
        plt.title("cost function across iterations")
        plt.legend()

        print "Plotted learning rate to "+ filename
        plt.savefig(filename)
        
    ################################################################################


    def model_size_comp(self, model_sizes, val_errors, train_errors, filename):
        fig = plt.figure(figsize=(16,10))
        plt.plot(model_sizes, val_errors, linewidth=1.0, label='validation accuracy')
        plt.plot(model_sizes, train_errors, linewidth=1.0, label='training accuray')
        plt.ylabel('accuracy')
        plt.xlabel('model size (r**2, r)')
        plt.axhline(0, color='black', linewidth=0.5)
    
        plt.legend(frameon=False)
        
        plt.savefig(filename)

        
    def plot_activation(self, nactivation1, nactivation2, filename):

        na1, na2 = [], []
        
        for r in nactivation1.T:
            na1.append(r.mean())

        for r in nactivation2.T:
            na2.append(r.mean())

        fig, ax1 = plt.subplots(figsize=(16,10))
        
        plt.subplot(211)
        ind = np.arange(len(na1))
        rects1 = plt.bar(ind, na1, color='y')

        plt.subplot(212)
        ind = np.arange(len(na2))
        rects1 = plt.bar(ind, na2, color='r')

        print "Plotted neuron activations to "+filename 
        plt.savefig(filename)


    ################################################################################

    def plot_nn_perf(self, history, filename):
        fig, ax1 = plt.subplots(figsize=(12,8))
        
        # Get training and test accuracy histories
        training_accuracy = history.history['acc']
        test_accuracy = history.history['val_acc']

        training_loss = history.history['loss']
        test_loss = history.history['val_loss']
        
        # Create count of the number of epochs
        epoch_count = range(1, len(training_accuracy) + 1)

        # Visualize accuracy history
        lns1 = ax1.plot(epoch_count, training_accuracy, 'r--', label='Training accuray')
        lns2 = ax1.plot(epoch_count, test_accuracy, 'b--', label='Validation accuracy')
        ax1.set_ylabel('Accuracy Score')
        
        ax2 = ax1.twinx()
        lns3 = ax2.plot(epoch_count, training_loss, 'y-', label='Training loss')
        lns4 = ax2.plot(epoch_count, test_loss, 'g-', label='Validation loss')
        ax2.set_ylabel('Loss')

        lns = lns1+lns2+lns3+lns4
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0, frameon=False)
        # plt.legend(['Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss'])
        plt.xlabel('Epoch')        

        plt.savefig(filename)

    def plot_model(self, model, filename):
        print("Plotted model structure to "+filename)
        plot_model(model, to_file=filename, show_shapes=True)
        

    
