import numpy as np
import matplotlib.pyplot as plt
import data
import functions


n = 100
m = 2
min_k = 0.5
max_k = 10
step_k = 0.5

# create h
h = np.ones((n, m))/np.sqrt(m)
data, labels = data.get_toy_dataset(n, m)

# calculate values of the objective
ks = []
objs = []
for i in range(int((max_k-min_k)/step_k)):
    new_k = min_k + i * step_k
    ks.append(new_k)
    objs.append(functions.ex4_objective(data, labels, new_k, h))

plt.plot(ks, objs, 'go--')