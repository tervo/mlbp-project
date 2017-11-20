# -*- coding: utf-8 -*-
import sys, re, numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import lib.io

# export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH

def main(argv):

    X, y = [], []

    io = lib.io.IO()
    
    # Read data
    X, y = io.read_data()    
    X,y = numpy.matrix(X), numpy.matrix(y).T
    print X.shape
    print y.shape

    sys.exit()
        
    plt.n, bins, patches = plt.hist(errors)

    plt.ylabel('Frequency')
    plt.xlabel('MSE')

    # Save figure
    plt.savefig('xx.png')    

if __name__ == "__main__":
    main(sys.argv[1:])

