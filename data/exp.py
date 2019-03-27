import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pickle

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
# from sklearn import metrics
# >>> labels_true = [0, 0, 0, 1, 1, 1]
# >>> labels_pred = [0, 0, 1, 1, 2, 2]

# >>> metrics.adjusted_rand_score(labels_true, labels_pred)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.metrics import roc_curve, f1_score, silhouette_score
from sklearn import manifold
from sklearn import linear_model
from sklearn.model_selection import train_test_split


def my_Kmeans(x, y, k=4, time=10, return_NMI=False):

    x = np.array(x)
    x = np.squeeze(x)
    y = np.array(y)

    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)

    estimator = KMeans(n_clusters=k)
    ARI_list = []  # adjusted_rand_score(
    NMI_list = []
    silhouette_score_list = []
    if time:
        for i in range(time):
            estimator.fit(x, y)
            y_pred = estimator.predict(x)
            score = normalized_mutual_info_score(y, y_pred)
            NMI_list.append(score)
            s2 = adjusted_rand_score(y, y_pred)
            ARI_list.append(s2)
            # silhouette_score
            labels = estimator.labels_
            s3 = silhouette_score(x, labels, metric='euclidean')
            silhouette_score_list.append(s3)
        # print('NMI_list: {}'.format(NMI_list))
        score = sum(NMI_list) / len(NMI_list)
        s2 = sum(ARI_list) / len(ARI_list)
        s3 = sum(silhouette_score_list) / len(silhouette_score_list)
        print('NMI (10 avg): {:.4f} , ARI (10avg): {:.4f}, silhouette(10avg): {:.4f}'.format(score, s2, s3))

    else:
        estimator.fit(x, y)
        y_pred = estimator.predict(x)
        score = normalized_mutual_info_score(y, y_pred)
        print("NMI on all label data: {:.5f}".format(score))
    if return_NMI:
        return score

