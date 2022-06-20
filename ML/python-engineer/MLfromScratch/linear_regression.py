import numpy as np

GRAPH_DIR = "/home/vincxu/personal/git/vx-learn/ML/python-engineer/MLfromScratch/graphs/"


class LineaRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)

        # params initialization
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            # linear regression
            y_pred = np.dot(X, self.weights) + self.bias
            print("y_pred:", y_pred.shape)

            # gradient
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # update params
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            print("weights:", self.weights, "bias:", self.bias)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


def main():
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

    regressor = LineaRegression(lr=0.011, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    mse = mean_squared_error(y_test, predictions)
    print("MSE:", mse)

    # print(X_train.shape)
    # print(np.shape(X_train))
    # print(X_train)
    # plt.figure(figsize=(8, 6))
    # plt.scatter(X[:, 0], y, color="b", marker="o", s=10)
    # print(plt)
    # plt.savefig(f"{GRAPH_DIR}lr.png")

    y_pred_line = regressor.predict(X)
    cmap = plt.get_cmap("viridis")
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.savefig(f"{GRAPH_DIR}lr.png")


if __name__ == "__main__":
    main()
