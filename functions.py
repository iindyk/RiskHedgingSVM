import numpy as np
from scipy.optimize import minimize


# lower semideviation
def lsd(x, p=None):
    n = len(x)
    if p is None:
        p = np.ones(n) / n
    ex = np.dot(x, p)
    ret = 0
    for i in range(n):
        ret += (max(0, ex - x[i]) ** 2) * p[i]
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


def sd_risk_measure(x, k, p=None):
    n = len(x)
    assert k > 0
    if p is None:
        p = np.ones(n) / n
    return k * sd(x, p) - np.dot(x, p)


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
    for i in range(n):
        ret[i] = 1 - k * (ez - z[i]) / lsd(x, p)
    return ret


# risk identifier of the sd_risk_measure
def sd_rm_identifier(x, k, p=None):
    n = len(x)
    assert k > 0
    if p is None:
        p = np.ones(n) / n
    ex = np.dot(x, p)
    ret = np.zeros(n)
    for i in range(n):
        ret[i] = 1 - k * (x[i]-ex) / sd(x, p)
    return ret


def ex4_objective(data, labels, k, h):
    # find optimal w, b
    n, m = np.shape(data)
    w0 = np.ones(m + 1)
    w0[-1] = 0
    cons = {'type': 'ineq', 'fun': lambda w_:
            -lsd_risk_measure(np.array([(labels[i] * (np.dot(w_[:m], data[i]) + w_[-1]) - 1) for i in range(n)]), k)}
    res = minimize(lambda w_: np.dot(w_, w_)/2, w0,
                   method='trust-constr', options={'maxiter': 10000},
                   constraints=cons)
    print(res.message)
    w = res.x[:m]
    b = res.x[-1]
    print('w= ', w)
    print('b= ', b)
    con = lsd_risk_measure(np.array([(labels[i] * (np.dot(w, data[i]) + b) - 1) for i in range(n)]), k)
    print('cons= ', con)
    x = np.array([labels[i] * (np.dot(w, data[i]) + b) - 1 for i in range(n)])
    q = lsd_rm_identifier(x, k)
    ret = sum([labels[i]*np.dot(h[i], w)*q[i] for i in range(n)])/n
    return ret, res.success and con < 1e-5


def ex5_objective(data, labels, k, h):
    # find optimal w, b
    n, m = np.shape(data)
    w0 = np.ones(m + 1)
    w0[-1] = 0
    cons = {'type': 'ineq', 'fun': lambda w_:
    -sd_risk_measure(np.array([(labels[i] * (np.dot(w_[:m], data[i]) + w_[-1]) - 1) for i in range(n)]), k)}
    res = minimize(lambda w_: np.dot(w_, w_) / 2, w0,
                   method='trust-constr', options={'maxiter': 10000},
                   constraints=cons)
    print(res.message)
    w = res.x[:m]
    b = res.x[-1]
    print('w= ', w)
    print('b= ', b)
    con = sd_risk_measure(np.array([(labels[i] * (np.dot(w, data[i]) + b) - 1) for i in range(n)]), k)
    print('cons= ', con)
    x = np.array([labels[i] * (np.dot(w, data[i]) + b) - 1 for i in range(n)])
    q = sd_rm_identifier(x, k)
    ret = sum([labels[i] * np.dot(h[i], w) * q[i] for i in range(n)]) / n
    return ret, res.success and con < 1e-5
