import numpy as np
import matplotlib.pylab as plt


def sigmoid(y):
    y = np.clip(y, -500, 500)
    return 1.0 / (1.0 + np.exp(-y))


def softmax(y):
    y = np.clip(y, -500, 500)
    exp = np.exp(y)
    return exp / np.sum(exp, axis=1).reshape(-1, 1)


def log_loss(y, ypred):
    eps = 10 ** -15
    ypred = np.clip(ypred, eps, 1 - eps)
    number_of_samples = len(y)
    predicted_y_prob = ypred[range(number_of_samples), np.argmax(y, axis=1)]
    total_cost = -np.sum(np.log(predicted_y_prob))
    return total_cost / number_of_samples


def get_updated_weights(w, learning_rate, x, y, ypred):
    number_of_samples = len(y)
    der = - (learning_rate * np.dot((y - ypred).T, x))
    der /= number_of_samples
    w = w - der
    return w


def add_bias_term(x):
    return np.insert(x, 0, [1], axis=1)


def train(x, y, iteration=100, learning_rate=0.1):
    X = add_bias_term(x)  # add constant 1 to account for bias
    # print("Mean {}".format(np.mean(X)))
    # X = X - np.mean(X)
    # X = X / np.std(X)
    y_reshaped = []
    for i in y:
        if i == 0:
            y_reshaped.append([1, 0])
        else:
            y_reshaped.append([0, 1])
    y = np.array(y_reshaped)

    labels_count = len(np.unique(y))
    weight_matrix = np.full((labels_count, np.shape(X)[1]), 0.1)
    losses = []
    weights = [weight_matrix]
    # plt.ion()
    for i in range(iteration):
        ypred = predict(weight_matrix, X)
        loss = log_loss(y, ypred)
        if i % 10 == 0:
            print("Iteration ", i, " and loss ", loss)
            losses.append(loss)
            # plt.plot(i, loss)
            # plt.scatter(i, loss)
            # plt.show()
            # plt.pause(0.1)
        current_weight = get_updated_weights(weight_matrix, learning_rate, X, y, ypred)
        weights.append(current_weight)
        # weight_matrix = np.sum(weights, axis=0) / len(weights)
        weight_matrix = current_weight

    return weight_matrix, losses


def predict(weight_matrix, x_test):
    return softmax(x_test.dot(weight_matrix.T))


def accuracy(weight_matrix, test, y_test):
    y_reshaped = []
    for i in y_test:
        if i == 0:
            y_reshaped.append([1, 0])
        else:
            y_reshaped.append([0, 1])
    y_test = np.array(y_reshaped)
    test = add_bias_term(test)
    number_of_samples = np.shape(test)[0]
    prediction = predict(weight_matrix, test)
    successful_predictions = np.sum(np.argmax(prediction, axis=1) == np.argmax(y_test, axis=1))
    accuracy = (successful_predictions / number_of_samples) * 100
    print('Accuracy = ', accuracy)
    return accuracy
