import numpy as np
from scipy.optimize import minimize, minimize_scalar


# lower semideviation
def lsd(x, p=None):
    n = len(x)
    if p is None:
        p = np.ones(n) / n
    ex = np.dot(x, p)
    ret = 0
    for i in range(n):
        ret += ((max(0, ex - x[i])) ** 2) * p[i]
    assert ret != 0
    return np.sqrt(ret)


def sd(x, p=None):
    n = len(x)
    if p is None:
        p = np.ones(n) / n
    ex = np.dot(x, p)
    ret = 0
    for i in range(n):
        ret += ((ex - x[i]) ** 2) * p[i]
    assert ret != 0
    return np.sqrt(ret)


# coherent risk measure based on the lower semideviation
def lsd_risk_measure(x, k, p=None):
    n = len(x)
    assert k > 0
    if p is None:
        p = np.ones(n) / n
    return k * lsd(x, p) - np.dot(x, p)


# risk measure based on the standard deviation
def sd_risk_measure(x, k, p=None):
    n = len(x)
    assert k > 0
    if p is None:
        p = np.ones(n) / n
    return k * sd(x, p) - np.dot(x, p)


# conditional value-at-risk
def cvar(x, alpha, p=None):
    n = len(x)
    if p is None:
        p = np.ones(n) / n
    res = minimize_scalar(lambda c: -c + sum([max(c - x[i], 0) * p[i] for i in range(n)]) / alpha)
    #if not res.success:
    #    print('Error in CVaR calculation!')
    #    print(res.message)
    #    print(res.fun)
    return res.fun


# value-at-risk
def var(x, alpha):
    _x = np.array(x)
    _x.sort()
    n = len(x)
    return -_x[int(alpha*n)]


# risk identifier of the lsd_risk_measure
def lsd_rm_identifier(x, k, p=None):
    n = len(x)
    assert k > 0
    if p is None:
        p = np.ones(n) / n
    ex = np.dot(x, p)
    z = np.zeros(n)
    for i in range(n):
        z[i] = max(0, ex - x[i])
    ez = np.dot(z, p)
    ret = np.zeros(n)
    lsd_ = lsd(x, p)
    for i in range(n):
        ret[i] = 1 - k * (ez - z[i]) / lsd_
    return ret


# risk identifier of the sd_risk_measure
def sd_rm_identifier(x, k, p=None):
    n = len(x)
    assert k > 0
    if p is None:
        p = np.ones(n) / n
    ex = np.dot(x, p)
    ret = np.zeros(n)
    sd_ = sd(x, p)
    for i in range(n):
        ret[i] = 1 - k * (x[i] - ex) / sd_
    return ret


# risk identifier for CVaR
def cvar_identifier(x, alpha, p=None):
    n = len(x)
    assert alpha > 0
    if p is None:
        p = np.ones(n) / n
    q0 = np.ones(n)
    cons = {'type': 'eq', 'fun': lambda q: np.dot(q, p) - 1}
    bounds = [(0, 1 / alpha)] * n
    res = minimize(lambda q: sum([x[i] * q[i] * p[i] for i in range(n)]), q0,
                   method='trust-constr', options={'maxiter': 10000},
                   bounds=bounds, constraints=cons)
    if not res.success:
        print('Error calculating CVaR identifier!')
    return -res.fun, res.x


def ex4_objective(data, labels, k, h):
    # find optimal w, b
    n, m = np.shape(data)
    w0 = np.ones(m + 1)
    w0[-1] = 0
    cons = {'type': 'ineq', 'fun': lambda w_: 1 - np.dot(w_[:m], w_[:m])}
    res = minimize(lambda w_: lsd_risk_measure(np.array([(labels[i] * (np.dot(w_[:m], data[i]) + w_[-1]) - 1)
                                                         for i in range(n)]), k), w0,
                   method='trust-constr', options={'maxiter': 10000},
                   constraints=cons)
    print(res.message)
    w = res.x[:m]
    b = res.x[-1]
    print('w= ', w)
    print('b= ', b)
    con = 1 - np.dot(w, w)
    print('cons= ', con)
    x = np.array([labels[i] * (np.dot(w, data[i]) + b) - 1 for i in range(n)])
    q = lsd_rm_identifier(x, k)
    ret = sum([labels[i] * np.dot(h[i], w) * q[i] for i in range(n)]) / n
    return ret, res.success and con > -1e-5


def ex5_objective(data, labels, k, h):
    # find optimal w, b
    n, m = np.shape(data)
    w0 = np.ones(m + 1)
    w0[-1] = 0
    cons = {'type': 'ineq', 'fun': lambda w_: 1 - np.dot(w_[:m], w_[:m])}
    res = minimize(lambda w_: sd_risk_measure(np.array([(labels[i] * (np.dot(w_[:m], data[i]) + w_[-1]) - 1)
                                                        for i in range(n)]), k), w0,
                   method='trust-constr', options={'maxiter': 10000},
                   constraints=cons)
    print(res.message)
    w = res.x[:m]
    b = res.x[-1]
    print('w= ', w)
    print('b= ', b)
    con = 1 - np.dot(w, w)
    print('cons= ', con)
    x = np.array([labels[i] * (np.dot(w, data[i]) + b) - 1 for i in range(n)])
    q = sd_rm_identifier(x, k)
    ret = sum([labels[i] * np.dot(h[i], w) * q[i] for i in range(n)]) / n
    return ret, res.success and con > -1e-5


def ex6_objective(data, labels, alpha, h):
    # find optimal w, b
    n, m = np.shape(data)
    w0 = np.ones(m + 1)
    w0[-1] = 0
    cons = {'type': 'ineq', 'fun': lambda w_: 1 - np.dot(w_[:m], w_[:m])}
    res = minimize(lambda w_: cvar(np.array([(labels[i] * (np.dot(w_[:m], data[i]) + w_[-1]) - 1)
                                             for i in range(n)]), alpha), w0,
                   method='SLSQP', options={'maxiter': 10000},
                   constraints=cons)
    print(res.message)
    w = res.x[:m]
    b = res.x[-1]
    print('w= ', w)
    print('b= ', b)
    con = 1 - np.dot(w, w)
    print('cons= ', con)
    x = np.array([labels[i] * (np.dot(w, data[i]) + b) - 1 for i in range(n)])
    _, q = cvar_identifier(x, alpha)
    ret = sum([labels[i] * np.dot(h[i], w) * q[i] for i in range(n)]) / n
    return ret, res.success and con > -1e-5


def get_histogram(x, a, b, n_bins):
    ret = np.zeros(n_bins)
    h = (b-a)/n_bins
    bins = np.array([a+h*i for i in range(n_bins)])
    for xi in x:
        if xi <= a+h:
            ret[0] += 1
        elif xi > b-h:
            ret[-1] += 1
        else:
            ret[int((xi-a)/h)] += 1
    return ret/len(x), bins


def decompose_x(x, m, n):
    return x[:m], x[m], x[m + 1:m * n + m + 1], \
           x[m * n + m + 1:(m + 1) * n + m + 1], x[(m + 1) * n + m + 1:(m + 2) * n + m + 1]  # w, b, h, l, a


def approx_fun(x):
    return max(x, -1.0)


def class_obj_inf(w, b, h, dataset, labels, C):
    av = 0.0
    n, m = np.shape(dataset)
    dataset_inf = np.array(dataset) + np.transpose(np.reshape(h, (m, n)))
    for i in range(n):
        av += max(0, 1 - labels[i]*(np.dot(w, dataset_inf[i])+b))
    return C*av + np.dot(w, w)/2


def coeff_diff(w1, w2, b1, b2):
    w1_n = w1 / np.sqrt(np.dot(w1, w1) + b1 ** 2)
    b1_n = b1 / np.sqrt(np.dot(w1, w1) + b1 ** 2)
    w2_n = w2 / np.sqrt(np.dot(w2, w2) + b2 ** 2)
    b2_n = b2 / np.sqrt(np.dot(w2, w2) + b2 ** 2)
    return np.sqrt(np.dot(w1_n - w2_n, w1_n - w2_n) + (b1_n - b2_n) ** 2)


def adv_obj(x, dataset, labels):
    n = len(dataset)
    m = len(dataset[0])
    av = 0.0
    for i in range(0, n):
        av += approx_fun(labels[i] * (np.dot(x[:m], dataset[i]) + x[m]))
    return av


def adv_obj_gradient(x, dataset, labels):
    ret = []
    n, m = np.shape(dataset)
    for j in range(0, m):
        ret.append(
            sum([labels[i] * dataset[i][j] * (1.0 if labels[i] * (np.dot(x[:m], dataset[i]) + x[m]) > -1.0 else 0.0)
                 for i in range(0, n)]))  # with respect to w[j]
    ret.append(sum([labels[i] * (1.0 if labels[i] * (np.dot(x[:m], dataset[i]) + x[m]) > -1.0 else 0.0)
                    for i in range(0, n)]))  # with respect to b
    for i in range(0, (2 + m) * n):
        ret.append(0.0)  # with respect to h, l, a
    return np.array(ret)


def class_constr_inf_ineq_convex_cobyla(x, w_prev, dataset, labels, eps, C):
    ret = []
    n, m = np.shape(dataset)
    w, b, h, l, a = decompose_x(x, m, n)
    for i in range(0, n):
        ret.append(l[i])  # for cobyla only
        ret.append(C - l[i])  # for cobyla only
        ret.append(a[i])  # for cobyla only
        ret.append(
            labels[i] * (np.dot(w, dataset[i]) + np.dot(w_prev, [h[j * n + i] for j in range(0, m)]) + b) - 1 + a[i])
    ret.append(eps * n - np.dot(h, h))
    return np.array(ret)


def class_constr_inf_eq_convex(x, w_prev, l_prev, dataset, labels, C):
    ret = []
    n, m = np.shape(dataset)
    w, b, h, l, a = decompose_x(x, m, n)
    for j in range(0, m):
        ret.append(w[j] - sum([l_prev[i] * labels[i] * (dataset[i][j] + h[j * n + i]) for i in range(0, n)]))
    ret.append(sum([l[i] * labels[i] for i in range(0, n)]))
    for i in range(0, n):
        hi = [h[j * n + i] for j in range(0, m)]
        ret.append(l[i] - l_prev[i] * a[i] - l_prev[i] * labels[i] * (np.dot(w, dataset[i]) + np.dot(w_prev, hi) + b))
        ret.append(l_prev[i] * a[i] - C * a[i])
    return np.array(ret)
