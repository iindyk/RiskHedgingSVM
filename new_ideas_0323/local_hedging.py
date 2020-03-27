import numpy as np
from scipy.optimize import minimize, root_scalar
from sklearn import svm
import data
import functions

data, labels = data.get_diabetic_dataset()  # get_toy_dataset(100, 3)
n, m = np.shape(data)
k = 2
alphas = [0.1, 0.05]

_svc = svm.SVC(kernel='linear').fit(data, labels)
w0 = np.ones(m + 1)
w0[:m] = _svc.coef_[0]
w0[-1] = _svc.intercept_[0]

# create h
h = np.zeros((n, m))
for i in range(n):
    h[i, :] = w0[:m]
h = 0.1*np.linalg.norm(data)*h/np.linalg.norm(h)


def deriv(l):
    lambdas = [l, 1-l]
    # train the model on original data
    cons = {'type': 'ineq', 'fun': lambda w_: 1 - np.dot(w_[:m], w_[:m])}
    res = minimize(lambda w_: sum([lambdas[j] * functions.cvar(
            np.array([(labels[i] * (np.dot(w_[:m], data[i]) + w_[-1]) - 1) for i in range(n)]), alphas[j]) for j in
              range(k)]), w0, method='SLSQP', options={'maxiter': 10000}, constraints=cons)
    print(res.message)
    w = res.x[:m]
    b = res.x[-1]
    x = np.array([labels[i] * (np.dot(w, data[i]) + b) - 1 for i in range(n)])
    q = []
    for i in range(k):
        _, _q = functions.cvar_identifier(x, alphas[i])
        q.append(_q)
    return sum([labels[i]*np.dot(w, h[i])*(l*q[0][i]+(1-l)*q[1][i]) for i in range(n)])


# find l for local hedging
sol = root_scalar(deriv, bracket=[0., 1.], x0=0.5, fprime=False, method='brenth')
l_opt = sol.root
print('iterations: ', sol.iterations)
print('function calls: ', sol.function_calls)
print('optimal lambdas are: ', l_opt, ' and ', 1-l_opt)
