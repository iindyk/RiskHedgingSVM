import numpy as np
import matplotlib.pyplot as plt
import data
import functions
from random import randint


def plot_k(obj, data, labels, h, min_k, max_k, step_k, title):
    # calculate values of the objective
    ks = []
    objs = []
    prev_fail = False
    for i in range(int((max_k-min_k)/step_k)):
        new_k = min_k + i * step_k
        new_obj, success = obj(data, labels, new_k, h)
        print('k= ', new_k, 'optimization success= ', success)
        if success:
            ks.append(new_k)
            objs.append(new_obj)
        elif prev_fail:
            break
        else:
            prev_fail = True

    for x, y in zip(ks, objs):
        label = "{:.2f}".format(y)
        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     ha='center')  # horizontal alignment can be left, right or center
    plt.xlabel('kappa')
    plt.ylabel('f')
    plt.title(title)
    plt.plot(ks, objs, 'go--')
    plt.show()


if __name__ == '__main__':
    n = 1000
    m = 10
    min_k = 0.1
    max_k = 2
    step_k = 0.1
    #title = 'lower semideviation-based risk measure'
    #fun = functions.ex4_objective
    #title = 'standard deviation-based risk measure'
    #fun = functions.ex5_objective
    title = 'CVaR'
    fun = functions.ex6_objective

    # create h
    h = np.ones((n, m)) / np.sqrt(m)
    for _ in range(int(n*m/2)):
        i = randint(0, n-1)
        j = randint(0, m-1)
        h[i, j] = -1.0
    data, labels = data.get_toy_dataset(n, m)
    plot_k(fun, data, labels, h, min_k, max_k, step_k, title)
