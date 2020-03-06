import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import data
import functions
from sklearn.metrics import accuracy_score
from sklearn import svm


def find_lambdas(data, labels, h, alphas):
    n, m = np.shape(data)
    n_l = len(alphas)
    lambdas = np.ones(n_l) / n_l
    l0 = np.ones(n_l) / n_l
    w0 = np.ones(m + 1)
    w0[-1] = 0
    errs = []
    nit = 0
    maxit = 50
    while True:
        print('iteration #', nit)
        # calculate q with t=0
        cons = {'type': 'ineq', 'fun': lambda w_:
        -sum([lambdas[j] * functions.cvar(
            np.array([(labels[i] * (np.dot(w_[:m], data[i]) + w_[-1]) - 1) for i in range(n)]), alphas[j]) for j in
              range(n_l)])}
        res = minimize(lambda w_: np.dot(w_[:m], w_[:m]) / 2, w0,
                       method='COBYLA', options={'maxiter': 10000},
                       constraints=cons)
        print(res.message)
        w0 = res.x[:m]
        w0_norm = np.dot(w0[:m], w0[:m])
        b0 = res.x[-1]
        x0 = np.array([labels[i] * (np.dot(w0, data[i]) + b0) - 1 for i in range(n)])
        q0 = []
        for i in range(n_l):
            _, q = functions.cvar_identifier(x0, alphas[i])
            q0.append(q)

        # calculate q with t=1
        cons = {'type': 'ineq', 'fun': lambda w_:
        -sum([lambdas[j] * functions.cvar(
            np.array([(labels[i] * (np.dot(w_[:m], data[i] + h[i]) + w_[-1]) - 1) for i in range(n)]), alphas[j]) for j
              in range(n_l)])}
        res = minimize(lambda w_: np.dot(w_[:m], w_[:m]) / 2, w0,
                       method='COBYLA', options={'maxiter': 10000},
                       constraints=cons)
        print(res.message)
        w1 = res.x[:m]
        w1_norm = np.dot(w1[:m], w1[:m])
        b1 = res.x[-1]
        x1 = np.array([labels[i] * (np.dot(w1, data[i]+h[i]) + b1) - 1 for i in range(n)])
        q1 = []
        for i in range(n_l):
            _, q = functions.cvar_identifier(x1, alphas[i])
            q1.append(q)

        # calculate new lambdas
        a = np.array([[sum([labels[j]*(w1_norm*(data[j, i]+h[j, i])*q1[k][j]-w0_norm*data[j, i]*q0[k][j])
                            for j in range(n)])/n for k in range(n_l)] for i in range(m)])
        cons = [{'type': 'eq', 'fun': lambda l: sum(l)-1}]
        for i in range(m):
            cons.append({'type': 'eq', 'fun': lambda l: np.dot(l, a[i])})
        bounds = [(0, 1)] * n_l

        res = minimize(lambda l: np.dot(l-lambdas, l-lambdas),
                                 #sum([1000*np.square(np.dot(l, a[i])) for i in range(m)]),
                       lambdas,
                       method='trust-constr', options={'maxiter': 50000},
                       bounds=bounds, constraints=cons)
        err = np.linalg.norm(lambdas - res.x)
        errs.append(err)
        nit += 1
        print(res.message)
        if res.success:
            lambdas = np.array(res.x)
        else:
            print('skipping the iteration, resetting lambdas')
            lambdas = np.ones(n_l) / n_l
        if err < 1e-3 or nit > maxit:
            break
    # plot errors, print lambdas
    plt.plot(np.arange(nit), errs)
    plt.xlabel('iteration #')
    plt.ylabel('l2 error')
    cons_viol = abs(sum(lambdas)-1)
    for i in range(m):
        cons_viol += abs(np.dot(lambdas, a[i]))
    print('constraint violation= ', cons_viol)
    print('lambdas: ', lambdas)
    print('errors: ', errs)
    # calculate params on original data
    cons = {'type': 'ineq', 'fun': lambda w_:
    -sum([lambdas[j] * functions.cvar(
        np.array([(labels[i] * (np.dot(w_[:m], data[i]) + w_[-1]) - 1) for i in range(n)]), alphas[j]) for j in
          range(n_l)])}
    res = minimize(lambda w_: np.dot(w_[:m], w_[:m]) / 2, w0,
                   method='COBYLA', options={'maxiter': 10000},
                   constraints=cons)
    print(res.message)
    w0 = res.x[:m]
    b0 = res.x[-1]
    pred_orig = np.sign(np.array([np.dot(w0, data[i])+b0 for i in range(n)]))
    err_orig = 1 - accuracy_score(labels, pred_orig)
    print('Error of orig classifier on orig data= ', err_orig)
    # calculate params on infected data
    cons = {'type': 'ineq', 'fun': lambda w_:
    -sum([lambdas[j] * functions.cvar(
        np.array([(labels[i] * (np.dot(w_[:m], data[i] + h[i]) + w_[-1]) - 1) for i in range(n)]), alphas[j]) for j
          in range(n_l)])}
    res = minimize(lambda w_: np.dot(w_[:m], w_[:m]) / 2, w0,
                   method='COBYLA', options={'maxiter': 10000},
                   constraints=cons)
    print(res.message)
    w1 = res.x[:m]
    b1 = res.x[-1]
    pred_inf = np.sign(np.array([np.dot(w1, data[i])+b1 for i in range(n)]))
    err_inf = 1 - accuracy_score(labels, pred_inf)
    print('Error of inf classifier on orig data= ', err_inf)
    # svc on orig data
    svc_orig = svm.SVC(kernel='linear')
    svc_orig.fit(data, labels)
    pred_svc_orig = svc_orig.predict(data)
    err_svc_orig = 1 - accuracy_score(labels, pred_svc_orig)
    print('Error of orig svc on orig data= ', err_svc_orig)
    # svc on infected data
    svc_inf = svm.SVC(kernel='linear')
    svc_inf.fit(data+h, labels)
    pred_svc_inf = svc_inf.predict(data)
    err_svc_inf = 1 - accuracy_score(labels, pred_svc_inf)
    print('Error of infected svc on orig data= ', err_svc_inf)

    plt.show()


if __name__ == '__main__':
    n = 200
    m = 4
    k = 5
    alphas = [1-i/(3*k) for i in range(1, k+1)]
    # create h
    h = np.ones((n, m)) / np.sqrt(m)
    data, labels = data.get_toy_dataset(n, m, random_flips=0.05)
    find_lambdas(data, labels, h, alphas)
