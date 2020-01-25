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
    maxit = 10
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
                       method='trust-constr')
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


def get_adversarial_shift_alt(data, labels, dist):
    C = 1.0
    maxit = 100
    learning_rate = 1e-5
    eps = dist
    delta = 1e-2
    n, m = np.shape(data)
    svc = svm.SVC(kernel='linear', C=C).fit(data, labels)
    predicted_labels = svc.predict(data)
    err_orig = 1 - accuracy_score(labels, predicted_labels)
    print('err on orig is ' + str(int(err_orig * 100)) + '%')
    nit = 0
    w = svc.coef_[0][:]
    b = svc.intercept_
    obj = 1e10
    h = np.zeros(m * n)
    while nit < maxit:
        print('iteration ' + str(nit) + '; start: ' + str(datetime.datetime.now().time()))
        h_p = h[:]
        obj_p = obj
        grad = fn.adv_obj_gradient(list(w) + [b] + [0.0 for i in range((m + 2) * n)], data, labels)
        w = w - learning_rate * grad[:m]
        b = b - learning_rate * grad[m]
        con = {'type': 'ineq', 'fun': lambda x: n * eps - np.dot(x, x)}
        cons = [con]
        sol = minimize(lambda x: fn.class_obj_inf(w, b, x, data, labels, C), np.zeros(m * n), constraints=cons)
        if sol.success:
            h = sol.x[:]
        print('success: ' + str(sol.success))
        print('message: ' + str(sol.message))
        print('nfev= ' + str(sol.nfev))
        print('w= ' + str(w))
        print('b= ' + str(b))
        print('attack_norm= ' + str(100 * np.dot(h, h) // (n * eps)) + '%')
        obj = fn.adv_obj(list(w) + [b] + [0.0 for i in range((m + 2) * n)], data, labels)
        nit += 1
        dataset_inf = np.array(data) + np.transpose(np.reshape(h, (m, n)))
        svc = svm.SVC(kernel='linear', C=C).fit(dataset_inf, labels)
        if (obj_p - obj < delta and np.dot(h, h) >= n * eps - delta) or not sol.success \
                or fn.coeff_diff(w, svc.coef_[0], b, svc.intercept_) > delta:
            break

    dataset_inf = np.array(data) + np.transpose(np.reshape(h_p, (m, n)))
    indices = range(n)
    # define infected points for graph
    inf_points = []
    k = 0
    for i in indices:
        if sum([h_p[j * n + k] ** 2 for j in range(m)]) > 0.9 * eps:
            inf_points.append(dataset_inf[i])
        k += 1
    svc1 = svm.SVC(kernel='linear', C=C)
    svc1.fit(dataset_inf, labels)
    predicted_labels_inf_svc = svc1.predict(data)
    err_inf_svc = 1 - accuracy_score(labels, predicted_labels_inf_svc)
    print('err on infected dataset by svc is ' + str(int(100 * err_inf_svc)) + '%')
    predicted_labels_inf_opt = np.sign([np.dot(data[i], w) + b for i in range(0, n)])
    err_inf_opt = 1 - accuracy_score(labels, predicted_labels_inf_opt)
    print('err on infected dataset by opt is ' + str(int(100 * err_inf_opt)) + '%')
    return dataset_inf, labels
