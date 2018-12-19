import numpy as np
import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def lr(iter):
    return np.exp(-iter / 5)


def online_cluster(X, cluster, epochs=100):
    random_ints = np.random.randint(0, X.shape[1], cluster)
    B = X[:, random_ints]
    y = []
    loss = 0
    for epoch in range(epochs):
        for i in range(X.shape[1]):
            x = X[:, i]
            # calculate the distance
            distances = np.sum(np.square(B - x.reshape(-1, 1)), axis=0)
            b_win = np.argmin(distances)
            B[:, b_win] += lr(epoch) * (x - B[:, b_win])

    for i in range(X.shape[1]):
        x = X[:, i]
        # calculate the distance
        distances = np.sum(np.square(B - x.reshape(-1, 1)), axis=0)
        b_win = np.argmin(distances)
        y.append(b_win)
        loss += np.sum(np.square((B[:, b_win] - x)))

    loss /= X.shape[1]

    return y, loss