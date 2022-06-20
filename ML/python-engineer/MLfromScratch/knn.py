from collections import Counter

import numpy as np


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


class KNN:
    def __init__(self, k=3) -> None:
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all samples in X_train
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Get the k nearest examples
        idx = np.argsort(distances)[: self.k]
        # Get their labels
        k_nearest_labels = [self.y_train[i] for i in idx]
        # Get the most common label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


if __name__ == "__main__":
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    cmap = ListedColormap(["#FF0000", "#0000FF", "00FF00"])

    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    k = 50  # Number of neighbors
    clf = KNN(k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy(y_test, y_pred))
