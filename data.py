import numpy as np
from random import randint


# returns a dataset of points from [0, 1]^m
def get_toy_dataset(n, m, random_flips=0.0):
    dataset = np.random.uniform(-0.5, 0.5, (n, m))
    labels = []
    for i in range(n):
        labels.append(1.0 if sum(dataset[i, :]) > 0 else -1.0)
    labels = np.array(labels)
    # random attack
    for i in range(int(random_flips*n)):
        k = randint(0, n-1)
        labels[k] *= -1.0
    class1 = np.sum(labels == 1)
    print('dataset generated; class 1: ', class1, ' points, class -1: ', n-class1, ' points.')
    return dataset, labels
