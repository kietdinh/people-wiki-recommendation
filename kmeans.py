import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.base import BaseEstimator, ClusterMixin
from tdigest import tdigest


class KMeansCosine(BaseEstimator, ClusterMixin):

    def __init__(self, n_clusters, iter=300, metric='euclidean'):
        self.n_clusters = n_clusters
        self.iter = iter
        self.metric = metric

        self.labels = []
        self.centers = []

        self.X = None

    def init_centers(self):
        return self.X[np.random.choice(self.X.shape[0], self.n_clusters, replace=False)]

    def assign_labels(self, centers):
        if self.metric == 'cosine':
            D = cosine_distances(self.X, centers)
        else:
            D = euclidean_distances(self.X, centers)
        return np.argmin(D, axis=1)

    def update_centers(self, labels):
        centers = np.zeros((self.n_clusters, self.X.shape[1]))
        for k in range(self.n_clusters):
            Xk = self.X[labels == k, :]
            centers[k, :] = np.mean(Xk, axis=0)
        return centers

    @staticmethod
    def has_converged(centers, new_centers):
        return (set([tuple(a) for a in centers]) ==
                set([tuple(a) for a in new_centers]))

    def fit(self, X, y=None):
        self.X = X
        self.centers = [self.init_centers()]
        self.labels = []
        it = 0
        while it <= self.iter:
            self.labels.append(self.assign_labels(self.centers[-1]))
            new_centers = self.update_centers(self.labels[-1])
            if self.has_converged(self.centers[-1], new_centers):
                break
            self.centers.append(new_centers)
            it += 1

        self.labels = self.labels[-1]
        self.centers = self.centers[-1]

    def fit_predict(self, X, y=None):
        self.fit(X, y)

    def distance(self):
        dist = []

        for l in set(self.labels):
            x = self.X[self.labels == l, :]
            if self.metric == 'cosine':
                D = cosine_distances(x, self.centers)
            else:
                D = euclidean_distances(x, self.centers)

            for d in D:
                dist.append(d[l])

        return dist

    def score(self, compression=100, q=0.8):
        t = tdigest.TDigest(compression)
        results = []

        for l in set(self.labels):
            x = self.X[self.labels == l, :]
            if self.metric == 'cosine':
                D = cosine_distances(x, self.centers)
            else:
                D = euclidean_distances(x, self.centers)

            for d in D:
                t.add(d[l])

            results.append(t.percentile(q))

        return results

    def display(self):
        if self.X.shape[1] != 2:
            return

        for l in set(self.labels):
            x = self.X[self.labels == l, :]
            plt.plot(x[:, 0], x[:, 1], 'o', markersize=4, alpha=.8)

        plt.axis('equal')
        plt.plot()
        plt.show()
