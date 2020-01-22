import numpy as np
from random import randint


# returns a dataset of points from [0, 1]^m
def get_toy_dataset(n, m, random_flips=0.1):
    dataset = np.random.uniform(0, 1, (n, m))
    labels = []
    for i in range(n):
        labels.append(1 if sum(dataset[i, :]) > 0.5*m else -1)
    # random attack
    for i in range(int(random_flips*n)):
        k = randint(0, n-1)
        labels[k] *= -1
    return dataset[:int(0.5*n), :], labels[:int(0.5*n)], dataset[int(0.5*n):, :], labels[int(0.5*n):]
