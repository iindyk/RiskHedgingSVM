import numpy as np
import matplotlib.pyplot as plt
import data
import functions


n = 500
m = 10
min_k = 0.05
max_k = 2
step_k = 0.05

# create h
h = np.ones((n, m))/np.sqrt(m)
data, labels = data.get_toy_dataset(n, m)

# calculate values of the objective
ks = []
objs = []
for i in range(int((max_k-min_k)/step_k)):
    new_k = min_k + i * step_k
    new_obj, success = functions.ex4_objective(data, labels, new_k, h)
    print('k= ', new_k, 'optimization success= ', success)
    if success:
        ks.append(new_k)
        objs.append(new_obj)

plt.plot(ks, objs, 'go--')
plt.show()
