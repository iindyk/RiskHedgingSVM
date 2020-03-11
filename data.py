import numpy as np
import matplotlib.pyplot as plt
import functions as fn
import datetime
from random import randint
from scipy.optimize import minimize
from sklearn import svm
from sklearn.metrics import accuracy_score
from art.attacks import PoisoningAttackSVM
from art.classifiers import SklearnClassifier
import os
import pickle


# returns a dataset of points from [0, 1]^m
def get_toy_dataset(n, m, random_flips=0.1, horizontal=False):
    dataset = np.random.uniform(-10, 10, (n, m))
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


def get_breast_cancer_dataset():
    f = open("wdbc.data", "r")
    data = []
    labels = []
    for line in f:
        arr = line.split(',')[1:]
        if arr[0] == 'B':
            labels.append(1.)
        else:
            labels.append(-1.)
        data.append(np.array([float(s) for s in arr[1:]]))
    f.close()
    data = np.array(data)
    labels = np.array(labels)
    class0 = np.sum(labels == 1.)
    class1 = np.sum(labels == -1.)
    print('dataset generated; class 1: ', class0, ' points, class -1: ', class1, ' points.')
    return data, labels


def get_gait_freeze_dataset():
    data_files = os.listdir("dataset_fog_release/")
    data = []
    labels = []
    for f_path in data_files:
        f = open("dataset_fog_release/"+f_path, "r")

        for line in f:
            arr = line.split(' ')[1:]
            if arr[-1] == '1\n':
                labels.append(1.)
                data.append(np.array([float(s) for s in arr[:9]]))
            elif arr[-1] == '2\n':
                labels.append(-1.)
                data.append(np.array([float(s) for s in arr[:9]]))
        f.close()
    data = np.array(data)[:5000]
    labels = np.array(labels)[:5000]
    class0 = np.sum(labels == 1.)
    class1 = np.sum(labels == -1.)
    print('dataset generated; class 1: ', class0, ' points, class -1: ', class1, ' points.')
    print(data[:5])
    return data, labels


def get_diabetic_dataset():
    f = open("diabetic_retinopathy.arff", 'r')
    data = []
    labels = []
    for line in f:
        if line[0] in ('@', ' '):
            continue
        arr = line.split(',')
        if arr[-1] == '0\n':
            labels.append(1.)
        else:
            labels.append(-1.)
        data.append(np.array([float(s) for s in arr[:-1]]))
    f.close()
    data = np.array(data)
    labels = np.array(labels)
    class0 = np.sum(labels == 1.)
    class1 = np.sum(labels == -1.)
    print('dataset generated; class 1: ', class0, ' points, class -1: ', class1, ' points.')
    return data, labels


def get_parkinson_dataset():
    f = open("parkinson_dataset/train_data.txt", "r")
    data = []
    labels = []
    for line in f:
        arr = line.split(',')[1:]
        if arr[-1] == '0\n':
            labels.append(1.)
        else:
            labels.append(-1.)
        data.append(np.array([float(s) for s in arr[:-1]]))
    f.close()
    data = np.array(data)
    labels = np.array(labels)
    class0 = np.sum(labels == 1.)
    class1 = np.sum(labels == -1.)
    print('dataset generated; class 1: ', class0, ' points, class -1: ', class1, ' points.')
    return data, labels


def get_spectf_heart_dataset():
    f = open("spectf_heart_dataset/SPECTF.train", "r")
    data = []
    labels = []
    for line in f:
        arr = line.split(',')
        if arr[0] == '0':
            labels.append(1.)
        else:
            labels.append(-1.)
        data.append(np.array([float(s) for s in arr]))
    f.close()
    data = np.array(data)
    labels = np.array(labels)
    class0 = np.sum(labels == 1.)
    class1 = np.sum(labels == -1.)
    print(data[:5])
    print('dataset generated; class 1: ', class0, ' points, class -1: ', class1, ' points.')
    return data, labels


def save_to_pickle(data, labels, name):
    obj = {'data': data, 'labels': labels}
    with open(name+'.pickle', 'wb') as f:
        pickle.dump(obj, f)
        f.close()
    print('Data saved!')


def load_from_pickle(name):
    with open(name+'.pickle', 'rb') as f:
        obj = pickle.load(f)
        f.close()
    print('Data loaded!')
    return obj['data'], obj['labels']


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
        for j in range(1, m):
            ret[i, j] = data[i, j]**2
    diff = np.linalg.norm(data - ret)
    ret = (dist/diff)*ret + (1-dist/diff)*data
    return ret, labels


def get_random_concept_drift(data, labels, dist):
    ret = np.zeros_like(data)
    n, m = np.shape(data)
    rotate = np.multiply(np.random.uniform(-1, 1, (n, m)), data)
    diff = np.linalg.norm(data-rotate)
    ret = (dist/diff)*rotate + (1-dist/diff)*data
    return ret, labels


def get_covariate_shift(data, labels, dist):
    ret = np.zeros_like(data)
    n, m = np.shape(data)
    for i in range(n):
        for j in range(m):
            ret[i, j] = data[i, j] - dist/np.sqrt(m*n)#labels[i]*dist/np.sqrt(m*n)
    return ret, labels


def get_poisoning(data, labels, dist):
    n, m = np.shape(data)
    step = 0.1
    eps = 0.1
    num_pois = int(0.3*n)
    poisoning_indices = np.random.randint(low=0, high=n-1, size=num_pois)
    svc = svm.SVC(kernel='linear')
    classifier = SklearnClassifier(model=svc, clip_values=(0, 100))
    one_hot_labels = []
    for l in labels:
        if l == 1:
            one_hot_labels.append(np.array([1, 0]))
        else:
            one_hot_labels.append(np.array([0, 1]))
    one_hot_labels = np.array(one_hot_labels)
    classifier.fit(data, one_hot_labels)

    # evaluate classifier on benign data
    predictions = classifier.predict(data)
    err_orig = accuracy_score(labels, np.argmax(predictions, axis=1)*2-1)
    print('Error on benign data: {}%'.format(err_orig * 100))

    # create adversarial examples
    attack = PoisoningAttackSVM(classifier=classifier, step=step, eps=eps,
                                x_train=data[poisoning_indices],
                                y_train=one_hot_labels[poisoning_indices],
                                x_val=data[poisoning_indices],
                                y_val=one_hot_labels[poisoning_indices],
                                max_iter=100)
    pois_data = attack.generate(data[poisoning_indices, :], one_hot_labels[poisoning_indices, :])
    data_infected = np.array(data)
    for i in range(len(poisoning_indices)):
        data_infected[poisoning_indices[i]] = pois_data[i]

    # evaluate poisoned classifier on benign data
    svc1 = svm.SVC(kernel='linear')
    svc1.fit(data_infected, labels)
    predictions = svc1.predict(data)
    err_pois = 1 - accuracy_score(labels, predictions)
    print('Poisoned error on benign data: {}%'.format(err_pois * 100))
    return data_infected, labels


def get_adversarial_shift(data, labels, dist):
    n, m = np.shape(data)
    C = 1.0
    maxit = 100
    eps = dist/100
    delta = 1e-4
    maxnum = int(0.3*n)
    svc = svm.SVC(kernel='linear', C=C).fit(data, labels)
    predicted_labels = svc.predict(data)
    err_orig = 1 - accuracy_score(labels, predicted_labels)
    print('err on orig is ' + str(int(err_orig * 100)) + '%')
    dataset_trunc, labels_trunc, indices \
        = truncate_by_dist(data, labels, maxnum)
    n_t = len(dataset_trunc)
    print('number of closest chosen points is ' + str(n_t))
    w = np.zeros(m)
    l = np.zeros(n_t)
    nit = 0
    while nit < maxit:
        print('iteration ' + str(nit) + '; start: ' + str(datetime.datetime.now().time()))
        con1 = {'type': 'ineq', 'fun': fn.class_constr_inf_eq_convex,
                'args': [w, l, dataset_trunc, labels_trunc, C]}
        con2 = {'type': 'ineq',
                'fun': lambda x: -1 * fn.class_constr_inf_eq_convex(x, w, l, dataset_trunc, labels_trunc, C)}
        con3 = {'type': 'ineq', 'fun': fn.class_constr_inf_ineq_convex_cobyla,
                'args': [w, dataset_trunc, labels_trunc, eps, C]}
        cons = [con1, con2, con3]
        sol = minimize(fn.adv_obj, np.zeros((m + 2) * n_t + m + 1), args=(dataset_trunc, labels_trunc), constraints=cons,
                       options={'maxiter': 10000},
                       method='COBYLA')
        print('success: ' + str(sol.success))
        print('message: ' + str(sol.message))
        x_opt = sol.x[:]
        w, b, h, l, a = fn.decompose_x(x_opt, m, n_t)
        print('nfev= ' + str(sol.nfev))
        print('w= ' + str(w))
        print('b= ' + str(b))
        print('attack_norm= ' + str(100 * np.dot(h, h) // (n_t * eps)) + '%')
        if fn.adv_obj(x_opt, dataset_trunc, labels_trunc) <= sol.fun + delta \
                and max(fn.class_constr_inf_eq_convex(x_opt, w, l, dataset_trunc, labels_trunc, C)) <= delta \
                and min(fn.class_constr_inf_eq_convex(x_opt, w, l, dataset_trunc, labels_trunc, C)) >= -delta \
                and min(
            fn.class_constr_inf_ineq_convex_cobyla(x_opt, w, dataset_trunc, labels_trunc, eps, C)) >= -delta \
                and sol.success and np.dot(h, h) / n_t >= eps - 100 * delta:
            break
        nit += 1

    dataset_infected = []
    print('attack norm= ' + str(np.dot(h, h) / n))
    print('objective value= ' + str(sol.fun))
    k = 0
    for i in range(0, n):
        tmp = []
        if i in indices:
            for j in range(0, m):
                tmp.append(data[i][j] + h[j * n_t + k])
            k += 1
        else:
            tmp = data[i]
        dataset_infected.append(tmp)
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
        if (obj_p - obj < delta and np.dot(h, h) >= n * eps - delta) or not sol.success: #\
                #or fn.coeff_diff(w, svc.coef_[0], b, svc.intercept_) > delta:
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


def truncate_by_dist(dataset, labels, muxnum):
    n, m = np.shape(dataset)
    dist = np.zeros(n)
    # calculate the distance
    for i in range(n):
        dist[i] = abs(dataset[i, 1]-dataset[i, 0])/np.sqrt(2)
    indices = dist.argsort()[:muxnum]
    dataset_trunc = dataset[indices]
    labels_trunc = labels[indices]
    return dataset_trunc, labels_trunc, indices


def get_poisoning_turn(dataset, labels, dist):
    n, m = np.shape(dataset)
    turn = np.radians(20)
    maxnum = int(0.5*n)
    svc = svm.SVC(kernel='linear', C=1)
    svc.fit(dataset, labels)
    predicted_labels = svc.predict(dataset)
    err = 1 - accuracy_score(labels, predicted_labels)
    print('err on original dataset by svc is ' + str(int(100 * err)) + '%')
    _, _, indices = truncate_by_dist(dataset, labels, maxnum)
    dataset_infected = np.array(dataset)
    for i in indices:
        dataset_infected[i, 0] = np.cos(turn)*dataset[i, 0]-np.sin(turn)*dataset[i, 1]
        dataset_infected[i, 1] = np.sin(turn)*dataset[i, 0]+np.cos(turn)*dataset[i, 1]
    svc1 = svm.SVC(kernel='linear', C=1)
    svc1.fit(dataset_infected, labels)
    predicted_labels_inf_svc = svc1.predict(dataset)
    err_inf_svc = 1 - accuracy_score(labels, predicted_labels_inf_svc)
    print('err on infected dataset by svc is ' + str(int(100 * err_inf_svc)) + '%')

    return dataset_infected, labels

