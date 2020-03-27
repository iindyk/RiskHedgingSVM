import numpy as np
from scipy.optimize import minimize, Bounds
import data
import functions
from sklearn import svm
from sklearn.metrics import accuracy_score

eps = 0.1
data, labels = data.get_breast_cancer_dataset()  # get_toy_dataset(100, 3)
n, m = np.shape(data)
k = m+1
alphas = [i/(k+1) for i in range(1, k+1)]
q = np.ones((k, n))
x = np.zeros(n)

# construct h
h = np.zeros((m, n, m))
h_norm = eps*np.linalg.norm(data)
for i in range(m):
    for j in range(n):
        h[i, j, i] = h_norm

_svc = svm.SVC(kernel='linear').fit(data, labels)
err_svc = 1 - accuracy_score(labels, _svc.predict(data))
print('l2 svc prediction error= ', err_svc)
w0 = np.ones(m + 1)
w0[:m] = _svc.coef_[0]
w0[-1] = _svc.intercept_[0]


def eq_constr(lambdas):
    # calculate q, w* and X*
    global x, q
    cons = {'type': 'ineq', 'fun': lambda w_: 1 - np.dot(w_[:m], w_[:m])}
    res = minimize(lambda w_: sum([lambdas[j] * functions.cvar(
        np.array([(labels[i] * (np.dot(w_[:m], data[i]) + w_[-1]) - 1) for i in range(n)]), alphas[j]) for j in
                                   range(k)]), w0, method='SLSQP', options={'maxiter': 10000}, constraints=cons)
    print(res.message)
    w = res.x[:m]
    b = res.x[-1]
    x = np.array([labels[i] * (np.dot(w, data[i]) + b) - 1 for i in range(n)])
    for i in range(k):
        _, _q = functions.cvar_identifier(x, alphas[i])
        q[i, :] = _q
    ret = []
    for _i in range(m):
        _ret = 0
        for _j in range(k):
            for _e in range(n):
                _ret += lambdas[_j]*labels[_e]*np.dot(h[_i, _e, :], w)*q[_j, _e]/n
        ret.append(_ret)
    ret.append(sum(lambdas)-1)
    return ret


def ineq_constr(lambdas):
    ret = 1.-1e-3
    for _i in range(k):
        for _j in range(n):
            ret += lambdas[_i]*x[n]*q[_i, _j]
    return ret


# main optimization
cons = [{'type': 'ineq', 'fun': ineq_constr},
        {'type': 'eq', 'fun': eq_constr}]

bounds = Bounds(lb=[0.]*k, ub=[1.]*k, keep_feasible=True)
lambdas0 = np.array([1/k]*k)
sol = minimize(lambda lambdas: 1., lambdas0, bounds=bounds, constraints=cons,
               options={'maxiter': 1000, 'disp': True}, method='SLSQP')

print('success: ', sol.success)
print('message: ', sol.message)
print('solution: ', sol.x)

# prediction error
err_cvar = 1 - accuracy_score(labels, np.sign(x+1))
print('resulting classifier prediction error= ', err_cvar)
