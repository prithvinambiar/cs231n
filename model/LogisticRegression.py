import numpy as np
import matplotlib.pylab as plt


def sigmoid(y):
    y = np.clip(y, -500, 500)
    return 1.0 / (1.0 + np.exp(-y))


def log_loss(y, ypred):
    eps = 10 ** -15
    ypred = np.clip(ypred, eps, 1 - eps)
    total_cost = -np.sum(y * np.log(ypred) + (1-y) * np.log(1-ypred))
    return total_cost / (np.shape(ypred)[0] * np.shape(ypred)[1])


def get_updated_weights(w, learning_rate, x, y, ypred):
    der = learning_rate * np.dot((ypred - y).T, x) + (0.1 * 2 * w)
    der = der / np.shape(ypred)[0]
    w = w - (learning_rate * der)
    return w


def add_bias_term(x):
    return np.insert(x, 0, [1], axis=1)


def train(x, y, iteration=100, learning_rate=1):
    X = add_bias_term(x)  # add constant 1 to account for bias
    print("Mean {}".format(np.mean(X)))
    # X = X - np.mean(X)
    # X = X / np.std(X)
    labels_count = len(np.unique(y))
    weight_matrix = np.full((labels_count, np.shape(X)[1]), 0.1)
    losses = []
    weights = [weight_matrix]
    plt.ion()
    for i in range(iteration):
        ypred = sigmoid(X.dot(weight_matrix.T))
        loss = log_loss(y, ypred)
        if i % 5 == 0:
            print("Iteration ", i, " and loss ", loss)
            losses.append(loss)
            # plt.plot(i, loss)
            plt.scatter(i, loss)
            plt.show()
            plt.pause(0.1)
        current_weight = get_updated_weights(weight_matrix, learning_rate, X, y, ypred)
        weights.append(current_weight)
        weight_matrix = np.sum(weights, axis=0) / len(weights)

    return weight_matrix, losses


def predict(weight_matrix, x_test):
    return np.argmax(sigmoid(x_test.dot(weight_matrix.T)))


def accuracy(weight_matrix, test, y_test):
    test = add_bias_term(test)
    success = 0
    number_of_samples = np.shape(test)[0]
    for i in range(number_of_samples):
        predicted_class = predict(weight_matrix, test[i, :])
        actual_class = np.argmax(y_test[i])
        if predicted_class == actual_class:
            success += 1
    print(success)
    print(number_of_samples)
    return (success / number_of_samples) * 100
