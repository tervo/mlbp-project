# -*- coding: utf-8 -*-
import sys, re, itertools
import numpy as np
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from keras.utils import plot_model
from keras import activations
from sklearn.metrics import confusion_matrix

class Viz:

    def label_hist(self, y, filename):
        print("Plotting label histogram")
        fig = plt.figure(figsize=(8,5))
        plt.hist(y, normed=True, alpha=0.75)
        plt.xticks( np.arange(1,10) )
        plt.xlabel('Train set genres (labels)')        
        plt.ylabel('Probability')
        plt.grid(True)
        plt.savefig(filename)
        
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
            
        rects1 = ax1.bar(ind, y1, width, color='#5799c6', label='Logistic loss')

        ax2 = ax1.twinx()
        rects2 = ax2.bar(ind + width+0.05, y2, width, color='#ff6450', label='Prediction score')

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

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, 1.05),
                   ncol=3, fancybox=True, shadow=True)
        plt.savefig(filename)

        print "Saved method comparison chart to " + filename
        
    def cross_results(self, x, ll, acc, filename):
        
        fig, ax1 = plt.subplots(figsize=(16,10))

        plt.grid()        
        
        lns1 = ax1.plot(
            x,
            ll,
            c="#27ae61",
            label="Logistic loss")
        ax1.set_ylabel("Logistic Loss")
        
        ax2 = ax1.twinx()
        lns2 = ax2.plot(
            x,
            acc,
            c="#c1392b",
            label="Accuracy")
        ax2.set_ylabel("Accuracy")

        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper center', frameon=False)
        ax1.set_xlabel("Components")

        print "Plotted cross validation results to "+ filename
        plt.savefig(filename)

            
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


    def plot_confusion_matrix(self, y_true, y_pred, classes,
                              normalize=False,
                              cmap=plt.cm.Blues,
                              filename='confusion_matrix.png'):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        fig, ax = plt.subplots()
        np.set_printoptions(precision=2)
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
            
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        tick_marks = np.arange(len(classes))
        ax.xaxis.tick_top()
        plt.xticks(tick_marks, classes) #, rotation=45)
        plt.yticks(tick_marks, classes)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
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
        
        
    def plot_feature_map(self, model, layer_id, X, n=256, ax=None, **kwargs):
        """
        """
        import keras.backend as K
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        layer = model.layers[layer_id]
        
        #try:
        get_activations = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
        activations = get_activations([X, 0])[0]
        #except:
            # Ugly catch, a cleaner logic is welcome here.
         #   raise Exception("This layer cannot be plotted.")
        
        # For now we only handle feature map with 4 dimensions
        if activations.ndim != 4:
            print("Activation dimension: %d"%activations.ndim)
            raise Exception("Feature map of '{}' has {} dimensions which is not supported.".format(layer.name,
                                                                                             activations.ndim))
        
        # Set default matplotlib parameters
        if not 'interpolation' in kwargs.keys():
            kwargs['interpolation'] = "none"

        if not 'cmap' in kwargs.keys():
            kwargs['cmap'] = "gray"
        
        fig = plt.figure(figsize=(15, 15))
    
        # Compute nrows and ncols for images
        n_mosaic = len(activations)
        nrows = int(np.round(np.sqrt(n_mosaic)))
        ncols = int(nrows)
        if (nrows ** 2) < n_mosaic:
            ncols +=1
        
        # Compute nrows and ncols for mosaics
        if activations[0].shape[0] < n:
            n = activations[0].shape[0]
    
        nrows_inside_mosaic = int(np.round(np.sqrt(n)))
        ncols_inside_mosaic = int(nrows_inside_mosaic)

        if nrows_inside_mosaic ** 2 < n:
            ncols_inside_mosaic += 1

        for i, feature_map in enumerate(activations):
            mosaic = make_mosaic(feature_map[:n], nrows_inside_mosaic, ncols_inside_mosaic, border=1)

            ax = fig.add_subplot(nrows, ncols, i+1)
        
            im = ax.imshow(mosaic, **kwargs)
            ax.set_title("Feature map #{} \nof layer#{} \ncalled '{}' \nof type {} ".format(i, layer_id,
                                                                                            layer.name,
                                                                                            layer.__class__.__name__))

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)
            
        fig.tight_layout()
        return fig


    def plot_all_feature_maps(self, model, X, file_prefix, n=256, ax=None, **kwargs):
        """
        """

        n = len(model.layers)
        r = n // 2
        c = n-r
        fig, axs = plt.subplots(r, c, figsize=(15, 6))
        figs = []
    
        for i, layer in enumerate(model.layers):
            
            try:
                fig = self.plot_feature_map(model, i, X, n=n, ax=ax, **kwargs)
                axs[i].imshow(fig)
            except Exception, e:
                print("failed %d"%i)
                print e
            else:
                figs.append(fig)

        fig.savefig(file_prefix+'feature_map.png')
        return figs
        
