# -*- coding: utf-8 -*-
import sys, re
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.externals import joblib

import random

class IO:
    
    def write_classes(self, filename, ids, data):
        """ Write results to file """

        file = open(filename, 'w')
        file.write("Sample_id,Sample_label\n")
        i = 0
        for line in data:
            file.write(str(ids[i]) + ',' + str(int(line))+'\n')
            i += 1
            
        file.close()
        
    def write_probabilities(self, filename, ids, data):
        """ Write results to file """

        file = open(filename, 'w')
        file.write("Sample_id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9,Class_10\n")
        i = 0
        for line in data:
            l = [ '%.4f' % elem for elem in line ]
            file.write(str(ids[i]) + ',' + ','.join(l)+'\n')
            i += 1
            
        file.close()

    def add_dim(self, x, value=1):
        new = []
        for row in x:
            new_row = []
            for i in x:
                new_row.append([i,value])
            new.append(new_row)
        return new
        
    def read_data(self, xfilename, yfilename):
        """ Read data from files """
        X, y = [], []
        
        # Read data    
        with open(xfilename) as f: lines = f.read().splitlines()
        for line in lines:
            l = map(float, line.split(','))
            X.append(l)
        
        try:
            with open(yfilename) as f: lines = f.read().splitlines()
            for line in lines: y.append(float(line))
        except:
            return X
        
        return X, y
    
    def split_data(self, x, y, num_of_batches, num_of_samples=None):
        """ Split data to train and validation data set 
        x:               [matrix, n,k]  feature list (samples as rows, features as cols)
        y:               [matrix, n,1]  label list
        num_of_batches:  [int]        number of batches to make
        num_of_samples:  [int]        number of samplest to pick to every batch (if None, all are taken)
        """

        selected, ids, batches_x, batches_y = [], [], [], []
        r = random.SystemRandom()
        k = len(x)
        batch_size = k/num_of_batches
        if num_of_samples is not None:
            if batch_size > num_of_samples:
                batch_size = num_of_samples
                
        for batch_num in range(num_of_batches):
            batch_x, batch_y, batch_ids = [], [], []
        
            while len(batch_x) < batch_size and len(x) > 0:
                i = r.randrange(0, len(x))            
                batch_x.append(x.pop(i))
                batch_y.append(y.pop(i))
                batch_ids.append(i)
            
            batches_x.append(np.matrix(batch_x))
            batches_y.append(np.matrix(batch_y).T)
            ids.append(batch_ids)
            
        return ids, batches_x, batches_y

    def pick_set(self, x, y, num_of_samples):
        """ Split data to train and validation data set 
        x:               [list, n,k]  feature list (samples as rows, features as cols)
        y:               [list, n,1]  label list
        num_of_samples:  [int]        number of samplest to pick
        """

        selected, ids, set_x, set_y = [], [], [], []
        r = random.SystemRandom()

        if len(set_x) >= num_of_samples:
            num_of_samples = len(set_x)-1
        
        while len(set_x) < num_of_samples:
            i = r.randrange(0, len(x))            
            set_x.append(x.pop(i))
            set_y.append(y.pop(i))
            ids.append(i)
            
        set_x = np.matrix(set_x)
        set_y = np.matrix(set_y).T
            
        return ids, set_x, set_y

    def shift_v(self, v, shift=-1):
        new_v = []
        for i in v:
            new_v.append(i+shift)
        return new_v

    def save_model(self, model, filename):
        joblib.dump(model, filename)
        print "Saved model to " + filename

    def load_model(self, filename):
        print "Loading model from " + filename
        return joblib.load(filename) 

