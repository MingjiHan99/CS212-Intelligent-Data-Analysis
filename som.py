import numpy as np
import somoclu
import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def sigma(init, iter):
    return init * np.exp(-iter / 2)


def h(i, j, sigma):
    return np.exp(-np.dot(i - j, i - j) / (sigma * sigma))


# learning rate
def lr(iter):
    return np.exp(-iter / 3)


def som(X, dim, iters=50):
    num_vectors = 1
    for i in dim:
        num_vectors *= i
    y = []
    randint = np.random.randint(0, X.shape[1], num_vectors)
    B = X[:, randint]
    dis_mat = np.zeros((B.shape[1], B.shape[1]), dtype=np.float32)

    for iter in range(iters):
        for i in range(X.shape[1]):
            x = X[:, i]
            distances = np.square(np.sum(B - x.reshape(-1, 1), axis=0))
            win = np.argmin(distances)
            for j in range(B.shape[1]):
                B[:, j] = B[:, j] + lr(iter) * h(B[:, win], B[:, j], sigma(2, iter)) * (x - B[:, j])

    for i in range(X.shape[1]):
        x = X[:, i]
        distances = np.square(np.sum(B - x.reshape(-1, 1), axis=0))
        win = np.argmin(distances)
        y.append(win)

    return B, y

def som_visual(X,y):
    n_rows, n_columns = 25, 25
    data = X.T
    color = []
    for i in range(len(y)):
        if y[i] == 1:
            color.append("red")
        else:
            color.append("blue")
    labels = range(569)
    som = somoclu.Somoclu(n_columns, n_rows, data=data)
    som.train()
    som.view_umatrix(bestmatches=True, bestmatchcolors=color, labels=labels)
    plt.savefig('./som/umatirx.jpg')