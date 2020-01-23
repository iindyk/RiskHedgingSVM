import numpy as np
import matplotlib.pyplot as plt
import data
import functions


def plot_k(obj, data, labels, h, min_k, max_k, step_k, title):
    # calculate values of the objective
    ks = []
    objs = []
    for i in range(int((max_k-min_k)/step_k)):
        new_k = min_k + i * step_k
        new_obj, success = obj(data, labels, new_k, h)
        print('k= ', new_k, 'optimization success= ', success)
        if success:
            ks.append(new_k)
            objs.append(new_obj)

    plt.xlabel('kappa')
    plt.ylabel('f')
    plt.title(title)
    plt.plot(ks, objs, 'go--')
    plt.show()


if __name__ == '__main__':
    n = 1000
    m = 10
    min_k = 0.5
    max_k = 1
    step_k = 0.05
    #title = 'lower semideviation-based risk measure'
    #fun = functions.ex4_objective
    #title = 'standard deviation-based risk measure'
    #fun = functions.ex5_objective
    title = 'CVaR'
    fun = functions.ex6_objective

    # create h
    h = np.ones((n, m)) / np.sqrt(m)
    data, labels = data.get_toy_dataset(n, m)
    plot_k(fun, data, labels, h, min_k, max_k, step_k, title)
