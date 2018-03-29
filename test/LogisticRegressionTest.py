from model import LogisticRegression
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


train_data = unpickle('../data/train')
test_data = unpickle('../data/test')
labels = unpickle('../data/meta')
interesting_coarse_labels = [0, 1]  # Aquatic mammals and Fish

train = []
y = []
test = []
y_test = []
for i in range(len(train_data[b'coarse_labels'])):
    for j in interesting_coarse_labels:
        if train_data[b'coarse_labels'][i] == j:
            train.append(train_data[b'data'][i])
            y.append(j)
            break

for i in range(len(test_data[b'coarse_labels'])):
    for j in interesting_coarse_labels:
        if test_data[b'coarse_labels'][i] == j:
            test.append(test_data[b'data'][i])
            y_test.append(j)
            break

train = np.array(train)
y = np.array(y)
test = np.array(test)
y_test = np.array(y_test)

y_reshaped = []
for i in y:
    if i == 0:
        y_reshaped.append([1, 0])
    else:
        y_reshaped.append([0, 1])
y_reshaped = np.array(y_reshaped)
weight_matrix, losses = LogisticRegression.train(train, y_reshaped,
                                                 iteration=10, learning_rate=0.1)
test_y_reshaped = []
for i in y_test:
    if i == 0:
        test_y_reshaped.append([1, 0])
    else:
        test_y_reshaped.append([0, 1])
test_y_reshaped = np.array(test_y_reshaped)
LogisticRegression.accuracy(weight_matrix, test, test_y_reshaped)
