import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import data
import functions


def find_lambdas(data, labels, h, alphas):
    n, m = np.shape(data)
    n_l = len(alphas)
    lambdas = np.ones(n_l) / n_l
    l0 = np.ones(n_l) / n_l
    w0 = np.ones(m + 1)
    w0[-1] = 0
    errs = []
    nit = 0
    maxit = 1
    while True:
        print('iteration #', nit)
        # calculate q with t=0
        cons = {'type': 'ineq', 'fun': lambda w_:
        -sum([lambdas[j] * functions.cvar(
            np.array([(labels[i] * (np.dot(w_[:m], data[i]) + w_[-1]) - 1) for i in range(n)]), alphas[j]) for j in
              range(n_l)])}
        res = minimize(lambda w_: np.dot(w_[:m], w_[:m]) / 2, w0,
                       method='SLSQP', options={'maxiter': 10000},
                       constraints=cons)
        print(res.message)
        w = res.x[:m]
        b = res.x[-1]
        x = np.array([labels[i] * (np.dot(w, data[i]) + b) - 1 for i in range(n)])
        q0 = []
        for i in range(n_l):
            _, q = functions.cvar_identifier(x, alphas[i])
            q0.append(q)

        # calculate q with t=1
        cons = {'type': 'ineq', 'fun': lambda w_:
        -sum([lambdas[j] * functions.cvar(
            np.array([(labels[i] * (np.dot(w_[:m], data[i] + h[i]) + w_[-1]) - 1) for i in range(n)]), alphas[j]) for j
              in range(n_l)])}
        res = minimize(lambda w_: np.dot(w_[:m], w_[:m]) / 2, w0,
                       method='SLSQP', options={'maxiter': 10000},
                       constraints=cons)
        print(res.message)
        w = res.x[:m]
        b = res.x[-1]
        x = np.array([labels[i] * (np.dot(w, data[i]) + b) - 1 for i in range(n)])
        q1 = []
        for i in range(n_l):
            _, q = functions.cvar_identifier(x, alphas[i])
            q1.append(q)

        # calculate new lambdas
        a = np.array([[sum([labels[j]*((data[j, i]+h[j, i])*q1[k][j]-data[j, i]*q0[k][j]) for j in range(n)])/n for k in range(n_l)] for i in range(m)])
        cons = [{'type': 'eq', 'fun': lambda l: sum(l)-1}]
        for i in range(m):
            cons.append({'type': 'eq', 'fun': lambda l: np.dot(l, a[i])})
        bounds = [(0, 1)] * n_l

        res = minimize(lambda l: np.dot(l-lambdas, l-lambdas), l0,
                       method='trust-constr', options={'maxiter': 10000},
                       bounds=bounds, constraints=cons)
        err = np.linalg.norm(lambdas - res.x)
        errs.append(err)
        nit += 1
        if res.success:
            lambdas = np.array(res.x)
        else:
            print(res.message)
            print('skipping the iteration, resetting lambdas')
            lambdas = np.ones(n) / n
        if err < 1e-5 or nit > maxit:
            break
    # plot errors, print lambdas
    plt.plot(np.arange(nit), errs)
    plt.xlabel('iteration #')
    plt.ylabel('l2 error')
    plt.show()
    cons_viol = abs(sum(lambdas)-1)
    for i in range(m):
        cons_viol += abs(np.dot(lambdas, a[i]))
    print('constraint violation= ', cons_viol)
    print(lambdas)


if __name__ == '__main__':
    n = 100
    m = 5
    k = 8
    alphas = [i/k for i in range(1, k)]
    # create h
    h = np.ones((n, m)) / np.sqrt(m)
    data, labels = data.get_toy_dataset(n, m)
    find_lambdas(data, labels, h, alphas)
