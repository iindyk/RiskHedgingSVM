import data
import functions
import numpy as np


n = 100
m = 2
dist = 10

# get benign data
dataset, labels = data.get_toy_dataset(n, m, horizontal=True)
data.graph_dataset(dataset, labels, 'original data')

# get concept drift
data_concept_drift, labels = data.get_concept_drift(dataset, labels, dist)
data.graph_dataset(data_concept_drift, labels, 'concept drift')
print('concept drift distance=', np.linalg.norm(dataset-data_concept_drift))

# get covariate shift
data_covariate_shift, labels = data.get_covariate_shift(dataset, labels, dist)
data.graph_dataset(data_covariate_shift, labels, 'covariate shift')
print('covariate shift distance=', np.linalg.norm(dataset-data_covariate_shift))

# get poisoned dataset
data_infected, labels = data.get_adversarial_shift_alt(dataset, labels, dist)
data.graph_dataset(data_infected, labels, 'poisoning')
print('infected dataset distance=', np.linalg.norm(dataset-data_infected))