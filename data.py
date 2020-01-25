import numpy as np
import matplotlib.pyplot as plt
import functions as fn
import datetime
from random import randint
from scipy.optimize import minimize
from sklearn import svm
from sklearn.metrics import accuracy_score


# returns a dataset of points from [0, 1]^m
def get_toy_dataset(n, m, random_flips=0.1, horizontal=False):
    dataset = np.random.uniform(-1, 1, (n, m))
    labels = []
    for i in range(n):
        if horizontal:
            labels.append(1.0 if dataset[i, 1]-dataset[i, 0] > 0 else -1.0)
        else:
            labels.append(1.0 if sum(dataset[i, :]) > 0 else -1.0)
    labels = np.array(labels)
    # random attack
    for i in range(int(random_flips*n)):
        k = randint(0, n-1)
        labels[k] *= -1.0
    class1 = np.sum(labels == 1)
    print('dataset generated; class 1: ', class1, ' points, class -1: ', n-class1, ' points.')
    return dataset, labels


def graph_dataset(data, labels, title):
    # make colors
    colors = []
    for l in labels:
        if l == 1:
            colors.append((0, 0, 1))
        else:
            colors.append((1, 0, 0))
    plt.scatter([float(i[0]) for i in data], [float(i[1]) for i in data], c=colors, cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.show()


def get_concept_drift(data, labels, dist):
    ret = np.zeros_like(data)
    n, m = np.shape(data)
    for i in range(n):
        ret[i, 0] = data[i, 0]
        ret[i, 1] = ((data[i, 1]+1)**2 - 2)/2
    diff = np.linalg.norm(data - ret)
    ret = (dist/diff)*ret + (1-dist/diff)*data
    return ret, labels


def get_covariate_shift(data, labels, dist):
    ret = np.zeros_like(data)
    n, m = np.shape(data)
    for i in range(n):
        ret[i, 0] = data[i, 0] - labels[i]*dist/np.sqrt(n)
        ret[i, 1] = data[i, 1]
    return ret, labels


def get_adversarial_shift(data, labels, dist):
    n, m = np.shape(data)
    C = 1.0
    maxit = 100
    eps = dist/100
    delta = 1e-4
    svc = svm.SVC(kernel='linear', C=C).fit(data, labels)
    predicted_labels = svc.predict(data)
    err_orig = 1 - accuracy_score(labels, predicted_labels)
    print('err on orig is ' + str(int(err_orig * 100)) + '%')
    w = np.zeros(m)
    l = np.zeros(n)
    x_opt = np.zeros((m + 2) * n + m + 1)
    nit = 0
    while nit < maxit:
        print('iteration ' + str(nit) + '; start: ' + str(datetime.datetime.now().time()))
        con1 = {'type': 'ineq', 'fun': fn.class_constr_inf_eq_convex,
                'args': [w, l, data, labels, C]}
        con2 = {'type': 'ineq',
                'fun': lambda x: -1 * fn.class_constr_inf_eq_convex(x, w, l, data, labels, C)}
        con3 = {'type': 'ineq', 'fun': fn.class_constr_inf_ineq_convex_cobyla,
                'args': [w, data, labels, eps, C]}
        cons = [con1, con2, con3]
        sol = minimize(fn.adv_obj, x_opt, args=(data, labels), constraints=cons, options={'maxiter': 10000},
                       method='COBYLA')
        print('success: ' + str(sol.success))
        print('message: ' + str(sol.message))
        x_opt = sol.x[:]
        w, b, h, l, a = fn.decompose_x(x_opt, m, n)
        print('nfev= ' + str(sol.nfev))
        print('w= ' + str(w))
        print('b= ' + str(b))
        print('attack_norm= ' + str(100 * np.dot(h, h) // (n * eps)) + '%')
        if fn.adv_obj(x_opt, data, labels) <= sol.fun + delta \
                and max(fn.class_constr_inf_eq_convex(x_opt, w, l, data, labels, C)) <= delta \
                and min(fn.class_constr_inf_eq_convex(x_opt, w, l, data, labels, C)) >= -delta \
                and min(
            fn.class_constr_inf_ineq_convex_cobyla(x_opt, w, data, labels, eps, C)) >= -delta \
                and sol.success and np.dot(h, h) / n >= eps - 100 * delta:
            break
        nit += 1

    dataset_infected = data + h
    print('attack norm= ' + str(np.dot(h, h) / n))
    print('objective value= ' + str(sol.fun))
    svc1 = svm.SVC(kernel='linear', C=C)
    svc1.fit(dataset_infected, labels)
    predicted_labels_inf_svc = svc1.predict(data)
    err_inf_svc = 1 - accuracy_score(labels, predicted_labels_inf_svc)
    print('err on infected dataset by svc is ' + str(int(100 * err_inf_svc)) + '%')
    return dataset_infected, labels
