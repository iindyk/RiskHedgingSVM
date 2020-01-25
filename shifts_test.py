import data
import functions as fn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy.optimize import minimize

n = 1000
m = 2
dist = 6
_min = -4
_max = 4
n_bins = 20

# get benign data
dataset, labels = data.get_toy_dataset(n, m, horizontal=True)

# get concept drift
data_concept_drift, labels = data.get_concept_drift(dataset, labels, dist)
print('concept drift distance=', np.linalg.norm(dataset - data_concept_drift))

# get covariate shift
data_covariate_shift, labels = data.get_covariate_shift(dataset, labels, dist)
print('covariate shift distance=', np.linalg.norm(dataset - data_covariate_shift))

# get poisoned dataset
data_infected, labels = data.get_poisoning_turn(dataset, labels, dist)
print('infected dataset distance=', np.linalg.norm(dataset - data_infected))

# graph datasets
data.graph_dataset(dataset, labels, 'original data')
data.graph_dataset(data_concept_drift, labels, 'concept drift')
data.graph_dataset(data_covariate_shift, labels, 'covariate shift')
data.graph_dataset(data_infected, labels, 'poisoning')

# calculate optimal w and b
# svc = svm.SVC(kernel='linear').fit(dataset, labels)
# w = svc.coef_[0][:]
# b = svc.intercept_
alpha = 0.05
w0 = np.ones(m + 1)
w0[-1] = 0
cons = {'type': 'ineq', 'fun': lambda w_:
-fn.cvar(np.array([(labels[i] * (np.dot(w_[:m], dataset[i]) + w_[-1]) - 1) for i in range(n)]), alpha)}
res = minimize(lambda w_: np.dot(w_[:m], w_[:m]) / 2, w0,
               method='SLSQP', options={'maxiter': 10000, 'disp': True},
               constraints=cons)
print(res.message)
w = res.x[:m]
b = res.x[-1]
print('w= ', w)
print('b= ', b)
con = fn.cvar(np.array([(labels[i] * (np.dot(w, dataset[i]) + b) - 1) for i in range(n)]), alpha)
print('cons= ', con)
x = np.array([labels[i] * (np.dot(w, dataset[i]) + b) - 1 for i in range(n)])

# calculate x for each case
x_orig = np.array([labels[i] * (np.dot(w, dataset[i]) + b) - 1 for i in range(n)])
x_conc = np.array([labels[i] * (np.dot(w, data_concept_drift[i]) + b) - 1 for i in range(n)])
x_cova = np.array([labels[i] * (np.dot(w, data_covariate_shift[i]) + b) - 1 for i in range(n)])
x_pois = np.array([labels[i] * (np.dot(w, data_infected[i]) + b) - 1 for i in range(n)])

# get histograms
hist_orig, bins = fn.get_histogram(x_orig, _min, _max, n_bins)
hist_conc, _ = fn.get_histogram(x_conc, _min, _max, n_bins)
hist_cova, _ = fn.get_histogram(x_cova, _min, _max, n_bins)
hist_pois, _ = fn.get_histogram(x_pois, _min, _max, n_bins)

# plot histograms
plt.xlabel('X')
plt.ylabel('relative frequency')
plt.plot(bins, hist_orig, 'go--', label='original data')
plt.plot(bins, hist_conc, 'bo--', label='concept drift')
plt.plot(bins, hist_cova, 'yo--', label='covariate shift')
plt.plot(bins, hist_pois, 'ro--', label='poisoning')
plt.legend()
plt.show()
