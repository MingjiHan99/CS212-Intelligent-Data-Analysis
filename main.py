import numpy as np
import csv
from show_hist import show_histgram
from show_cov import show_cov
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cluster import online_cluster
from som import som
from som import som_visual
X = []
y = []
n = 0
if __name__=="__main__":
    csv_reader = csv.reader(open('cancer.csv'))

    for i in range(30):
        X.append([])

    data = [x for x in csv_reader]

    for i in range(1, len(data)):
        for j in range(1, 32):
            if j == 1:
                if data[i][j] == 'M':
                    y.append(1)
                else:
                    y.append(0)
            else:
                X[j - 2].append(float(data[i][j]))
    X = np.array(X)
    y = np.array(y)
    pos = 0
    neg = 0
    for element in y:
        if element == 1:
            pos += 1
        else:
            neg += 1

    show_histgram(X,y)

    X_mean = np.sum(X, axis=1) / X.shape[1]
    X_std_var = np.std(X, axis=1)
    X = (X - X_mean.reshape(-1, 1)) / X_std_var.reshape(-1, 1)

    w,V = show_cov(X)
    plt.figure()
    plt.title('Eigenvalue')
    plt.bar(x=range(0, len(w)), height=w)
    plt.savefig('./eign/eignvalue.jpg')
    plt.figure()
    ratio = []
    sum = 0
    for i in w:
        sum += i
        ratio.append(sum)
    ratio = np.array(ratio) / np.sum(w)
    plt.title('Ratio')
    plt.bar(x=range(30), height=ratio)
    plt.savefig('./eign/ratio.jpg')

    X_two = np.dot(V.T[:2], X)

    X_three = np.dot(V.T[:3], X)

    y_prediction, loss2 = online_cluster(X_two, 2, 150)
    plt.figure()
    plt.scatter(X_two.T[:, 0], X_two.T[:, 1], s=50, c=y_prediction, cmap='viridis')
    plt.savefig('./cluster/cluster2d.jpg')

    fig = plt.figure()
    ax = Axes3D(fig)
    y_prediction, loss3 = online_cluster(X_three, 2, 150)

    ax.scatter(X_three.T[:, 0], X_three.T[:, 1], X_three.T[:, 2], c=y_prediction, s=50, cmap='viridis')
    plt.savefig('./cluster/cluster3d.jpg')

    #Real Data
    plt.figure()
    plt.scatter(X_two.T[:, 0], X_two.T[:, 1], c=y, s=50, cmap='viridis')
    plt.savefig('./truth/truth2d.jpg')
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(X_three.T[:, 0], X_three.T[:, 1], X_three.T[:, 2], c=y, s=50, cmap='viridis')
    plt.savefig('./truth/truth3d.jpg')


    ##SOM

    length = 8
    width = 8

    code_book, prediction = som(X, [length, width])
    matrix = np.zeros((length * width), dtype=np.int32)
    for i in prediction:
        matrix[i] += 1

    matrix = matrix.reshape(length, width)
    axis_labels = [x for x in range(length)]

    fig, ax = plt.subplots()
    im = ax.imshow(matrix)
    ax.set_xticks(np.arange(len(axis_labels)))
    ax.set_yticks(np.arange(len(axis_labels)))
    ax.set_xticklabels(axis_labels)
    ax.set_yticklabels(axis_labels)

    for i in range(0, len(axis_labels)):
        for j in range(0, len(axis_labels)):
            ax.text(i, j, np.around(matrix, 1)[i, j], ha="center", va="center", color="w")

    plt.savefig('./som/som_distribution.jpg')
    print(y)
    som_visual(X,y)