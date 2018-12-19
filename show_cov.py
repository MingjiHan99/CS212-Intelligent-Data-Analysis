import numpy as np
import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

def show_cov(X):
    C = np.dot(X, X.T) / X.shape[1]

    w, V = np.linalg.eigh(C)

    w = np.array(w)
    w = w[::-1]

    V_new = np.zeros((X.shape[0], X.shape[0]), dtype=np.float32)
    for i in range(V_new.shape[0]):
        V_new[:, i] = V[:, V_new.shape[0] - i - 1]

    C_new = np.dot(np.dot(V_new.T, C), V_new)

    axis_labels = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
                   'concavity_mean',
                   'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
                   'perimeter_se', 'area_se',
                   'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                   'fractal_dimension_se', 'radius_worst', 'texture_worst',
                   'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
                   'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
                   ]
    ratio = []
    sum = 0
    for i in w:
        sum += i
        ratio.append(sum)
    ratio = np.array(ratio) / np.sum(w)
    fig, ax = plt.subplots(figsize=(11, 11))

    im = ax.imshow(C)
    ax.set_xticks(np.arange(len(axis_labels)))
    ax.set_yticks(np.arange(len(axis_labels)))
    ax.set_xticklabels(axis_labels)
    ax.set_yticklabels(axis_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(0, 30):
        for j in range(0, 30):
            ax.text(i, j, np.around(C, 1)[i, j], ha="center", va="center", color="w")

    plt.title('Covariance Matrix')

    plt.savefig("./cov/cov.jpg")

    return w,V_new