import numpy as np
import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

def show_histgram(X,y):
    labels = ['Malignant','Benign']
    #plt.title('Dataset')
    #plt.pie(size,labels=labels, autopct='%1.1f%%',pctdistance=0.8, shadow=True)
    X_M = X[0:10,y == 1]
    X_B= X[0:10,y == 0]
    axis_labels =['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean',
                  'concave points_mean','symmetry_mean','fractal_dimension_mean']
    print(X_M.shape)
    for i in range(10):
        plt.hist(X_M[i],bins=20,alpha=0.75,label='Malignant')
        plt.hist(X_B[i],bins=20,alpha=0.75,label='Benign')
        plt.title(axis_labels[i])
        plt.savefig('./hist/'+str(i)+'.png')
        plt.figure()

