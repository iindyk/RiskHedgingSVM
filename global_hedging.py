import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import data
import functions
from sklearn.metrics import accuracy_score
from sklearn import svm


def find_lambdas(data, labels, h, alphas, data_test, labels_test):
    n, m = np.shape(data)
    n_l = len(alphas)
    lambdas = np.ones(n_l) / n_l
    _svc = svm.SVC(kernel='linear').fit(data, labels)
    _w0 = np.ones(m + 1)
    _w0[:m] = _svc.coef_[0]
    _w0[-1] = _svc.intercept_[0]
    errs = []
    nit = 0
    maxit = 0
    while True:
        print('iteration #', nit)
        # calculate q with t=0
        cons = {'type': 'ineq', 'fun': lambda w_:
        -sum([lambdas[j] * functions.cvar(
            np.array([(labels[i] * (np.dot(w_[:m], data[i]) + w_[-1]) - 1) for i in range(n)]), alphas[j]) for j in
              range(n_l)])}
        res = minimize(lambda w_: np.dot(w_[:m], w_[:m]) / 2, _w0,
                       method='SLSQP', options={'maxiter': 10000},
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
        res = minimize(lambda w_: np.dot(w_[:m], w_[:m]) / 2, _w0,
                       method='SLSQP', options={'maxiter': 10000},
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
        print('err= ', err)
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
    res = minimize(lambda w_: np.dot(w_[:m], w_[:m]) / 2, _w0,
                   method='SLSQP', options={'maxiter': 10000},
                   constraints=cons)
    print(res.message)
    w0 = res.x[:m]
    b0 = res.x[-1]
    pred_orig = np.sign(np.array([np.dot(w0, data_test[i])+b0 for i in range(len(data_test))]))
    err_orig = 1 - accuracy_score(labels_test, pred_orig)
    print('Error of orig classifier on orig data= ', err_orig)
    # calculate params on infected data
    cons = {'type': 'ineq', 'fun': lambda w_:
    -sum([lambdas[j] * functions.cvar(
        np.array([(labels[i] * (np.dot(w_[:m], data[i] + h[i]) + w_[-1]) - 1) for i in range(n)]), alphas[j]) for j
          in range(n_l)])}
    res = minimize(lambda w_: np.dot(w_[:m], w_[:m]) / 2, _w0,
                   method='SLSQP', options={'maxiter': 10000},
                   constraints=cons)
    print(res.message)
    w1 = res.x[:m]
    b1 = res.x[-1]
    pred_inf = np.sign(np.array([np.dot(w1, data_test[i])+b1 for i in range(len(data_test))]))
    err_inf = 1 - accuracy_score(labels_test, pred_inf)
    print('Error of inf classifier on orig data= ', err_inf)
    # l2 svc on orig data
    svc_orig = svm.SVC(kernel='linear')
    svc_orig.fit(data, labels)
    pred_svc_orig = svc_orig.predict(data_test)
    err_svc_orig = 1 - accuracy_score(labels_test, pred_svc_orig)
    print('Error of orig l2 svc on orig data= ', err_svc_orig)
    # l2 svc on infected data
    svc_inf = svm.SVC(kernel='linear')
    svc_inf.fit(data+h, labels)
    pred_svc_inf = svc_inf.predict(data_test)
    err_svc_inf = 1 - accuracy_score(labels_test, pred_svc_inf)
    print('Error of inf l2 svc on orig data= ', err_svc_inf)
    # l1 svc on orig data
    svc_orig_l1 = svm.LinearSVC(penalty='l1', dual=False)
    svc_orig_l1.fit(data, labels)
    pred_svc_orig_l1 = svc_orig_l1.predict(data_test)
    err_svc_orig_l1 = 1 - accuracy_score(labels_test, pred_svc_orig_l1)
    print('Error of orig l1 svc on orig data= ', err_svc_orig_l1)
    # l1 svc on inf data
    svc_inf_l1 = svm.LinearSVC(penalty='l1', dual=False)
    svc_inf_l1.fit(data+h, labels)
    pred_svc_inf_l1 = svc_inf_l1.predict(data_test)
    err_svc_inf_l1 = 1 - accuracy_score(labels_test, pred_svc_inf_l1)
    print('Error of inf l1 svc on orig data= ', err_svc_inf_l1)
    # VaR-SVM on orig data
    alpha = 0.65
    cons = {'type': 'ineq', 'fun': lambda w_:
        -functions.var([(labels[i] * (np.dot(w_[:m], data[i]) + w_[-1])) for i in range(n)], alpha)-1}
    res = minimize(lambda w_: np.dot(w_[:m], w_[:m]) / 2, _w0,
                   method='trust-constr', options={'maxiter': 10000},
                   constraints=cons)
    print(res.message)
    w_var_orig = res.x[:m]
    b_var_orig = res.x[-1]
    pred_var_orig = np.sign(np.array([np.dot(w_var_orig, data_test[i]) + b_var_orig for i in range(len(data_test))]))
    err_var_orig = 1 - accuracy_score(labels_test, pred_var_orig)
    print('Error of orig VaR svc on orig data= ', err_var_orig)
    # VaR-SVM on infected data
    cons = {'type': 'ineq', 'fun': lambda w_:
    -functions.var([(labels[i] * (np.dot(w_[:m], data[i]+h[i]) + w_[-1])) for i in range(n)], alpha)-1}
    res = minimize(lambda w_: np.dot(w_[:m], w_[:m]) / 2, _w0,
                   method='trust-constr', options={'maxiter': 10000},
                   constraints=cons)
    print(res.message)
    w_var_inf = res.x[:m]
    b_var_inf = res.x[-1]
    pred_var_inf = np.sign(np.array([np.dot(w_var_inf, data_test[i]) + b_var_inf for i in range(len(data_test))]))
    err_var_inf = 1 - accuracy_score(labels_test, pred_var_inf)
    print('Error of inf VaR svc on orig data= ', err_var_inf)
    # nu-SVM on orig data
    res = minimize(lambda w_: np.dot(w_[:m], w_[:m]) / 2 -
                              (1-alpha)*functions.cvar([(labels[i] * (np.dot(w_[:m], data[i]) + w_[-1])-1) for i in range(n)], alpha), _w0,)
    print(res.message)
    w_nu_orig = res.x[:m]
    b_nu_orig = res.x[-1]
    pred_nu_orig = np.sign(np.array([np.dot(w_nu_orig, data_test[i]) + b_nu_orig for i in range(len(data_test))]))
    err_nu_orig = 1 - accuracy_score(labels_test, pred_nu_orig)
    print('Error of orig nu svc on orig data= ', err_nu_orig)
    # nu-SVM on inf data
    res = minimize(lambda w_: np.dot(w_[:m], w_[:m]) / 2 -
                              (1 - alpha) * functions.cvar(
        [(labels[i] * (np.dot(w_[:m], data[i]+h[i]) + w_[-1])-1) for i in range(n)], alpha), _w0, )
    print(res.message)
    w_nu_inf = res.x[:m]
    b_nu_inf = res.x[-1]
    pred_nu_inf = np.sign(np.array([np.dot(w_nu_inf, data_test[i]) + b_nu_inf for i in range(len(data_test))]))
    err_nu_inf = 1 - accuracy_score(labels_test, pred_nu_inf)
    print('Error of orig nu svc on orig data= ', err_nu_inf)

    plt.show()


if __name__ == '__main__':
    n = 383
    m = 19
    k = 20
    alphas = [1-i/(1.5*k) for i in range(1, k+1)]

    # data, labels = data.get_toy_dataset(n*3, m, random_flips=0.05)
    data, labels = data.get_diabetic_dataset()
    data, data_test = data[:n], data[n:]
    labels, labels_test = labels[:n], labels[n:]
    print('data norm=', np.linalg.norm(data) / len(data))

    # create h
    #h = 20*np.ones((n, m)) / np.sqrt(m)
    svc = svm.LinearSVC(penalty='l1', dual=False)
    svc.fit(data, labels)
    h = svc.coef_[0]
    _h = np.ones((n, m))
    for i in range(n):
        _h[i] = h
    h = _h*0.1*np.linalg.norm(data)/np.linalg.norm(_h)
    print('perturbation norm=', np.linalg.norm(h) / len(data))

    find_lambdas(data, labels, h, alphas, data_test, labels_test)
