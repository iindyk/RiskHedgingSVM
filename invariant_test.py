from sklearn import svm
from sklearn.metrics import accuracy_score
import data as dt
import functions as fn
import numpy as np
import pickle
from art.attacks import PoisoningAttackSVM
from art.classifiers import SklearnClassifier
from statsmodels import robust


data_name = 'artificial'
read = False

if not read:
    n = 100
    m = 10
    a = -10
    b_ = 10
    n_bins = 50
    n_pois = 50
    n_sample = 20
    data, labels = dt.get_toy_dataset(n, m, random_flips=0.05)
    dist = 0.1 * np.linalg.norm(data) * n_pois / len(data)
    # train svm
    svm_ = svm.SVC(kernel='linear').fit(data, labels)
    w = svm_.coef_[0]
    b = svm_.intercept_[0]
    # get normalized histogram
    x_norm = np.array([labels[i] * (np.dot(data[i], w) + b) / np.linalg.norm(w) for i in range(n)])
    hist_orig = fn.get_histogram(x_norm, a, b_, n_bins)[0]
    print('histogram of original data: ', hist_orig)
    data_stream_tr = [data]
    labels_stream_tr = [labels]
    data_detection_tr = [hist_orig]
    labels_detection_tr = [1]
    # generate poisonings
    classifier = SklearnClassifier(model=svm_, clip_values=(0, 100))
    one_hot_labels = []
    for l in labels:
        if l == 1:
            one_hot_labels.append(np.array([1, 0]))
        else:
            one_hot_labels.append(np.array([0, 1]))
    one_hot_labels = np.array(one_hot_labels)
    classifier.fit(data, one_hot_labels)

    data_detection_te = []
    labels_detection_te = []
    data_stream_te = []
    labels_stream_te = []
    target_sets = [data_detection_tr, data_detection_te, data_detection_te]
    target_stream_sets = [data_stream_tr, data_stream_te, data_stream_te]
    target_labels = [labels_detection_tr, labels_detection_te, labels_detection_te]
    target_stream_labels = [labels_stream_tr, labels_stream_te, labels_stream_te]

    for j in range(3):
        for i in range(n_pois):
            poisoning_indices = np.random.randint(0, n, n_pois)
            attack = PoisoningAttackSVM(classifier=classifier, step=0.1, eps=0.1,
                                        x_train=data,
                                        y_train=one_hot_labels,
                                        x_val=data[:n_pois],
                                        y_val=one_hot_labels[:n_pois],
                                        max_iter=100)
            pois_data = attack.generate(data[poisoning_indices, :], one_hot_labels[poisoning_indices, :])
            pois_labels = one_hot_labels[poisoning_indices, :]
            target_stream_sets[j].append(pois_data)
            target_stream_labels[j].append(pois_labels)
            pois_x = np.array([(1 if pois_labels[i, 0] == 1 else -1) *
                               (np.dot(pois_data[i], w)+b)/np.linalg.norm(w) for i in range(n_pois)])
            target_sets[j].append(fn.get_histogram(pois_x, a, b_, n_bins)[0])
            target_labels[j].append(-1)

        # generate covariate shifts
        for i in range(n_pois):
            indices = np.random.randint(0, n, n_pois)
            cov_shift_data = data[indices]
            cov_shift_labels = labels[indices]
            target_stream_sets[j].append(cov_shift_data)
            target_stream_labels[j].append(cov_shift_labels)
            cov_shift_x = np.array([cov_shift_labels[i] *
                                    (np.dot(cov_shift_data[i], w)+b)/np.linalg.norm(w) for i in range(n_pois)])
            target_sets[j].append(fn.get_histogram(cov_shift_x, a, b_, n_bins)[0])
            target_labels[j].append(1)

        # generate concept drifts
        for i in range(n_pois):
            indices = np.random.randint(0, n, n_pois)
            conc_drift_data, conc_drift_labels = dt.get_random_concept_drift(data[indices], labels[indices], dist)
            target_stream_sets[j].append(conc_drift_data)
            target_stream_labels[j].append(conc_drift_labels)
            conc_drift_x = np.array([conc_drift_labels[i] *
                                    (np.dot(conc_drift_data[i], w)+b)/np.linalg.norm(w) for i in range(n_pois)])
            target_sets[j].append(fn.get_histogram(conc_drift_x, a, b_, n_bins)[0])
            target_labels[j].append(1)
        if j == 0:
            print('training data generated!')
    print('test data generated!')
    # save to pickle
    obj = {'data_detection_tr': np.array(data_detection_tr), 'labels_detection_tr': np.array(labels_detection_tr),
           'data_detection_te': np.array(data_detection_te), 'labels_detection_te': np.array(labels_detection_te),
           'data_stream_tr': data_stream_tr, 'labels_stream_tr': labels_stream_tr,
           'data_stream_te': data_stream_te, 'labels_stream_te': labels_stream_te}
    with open('invariant_data/' + data_name + '.pickle', 'wb') as f:
        pickle.dump(obj, f)
        f.close()
    print('Data saved!')
else:
    with open('invariant_data/' + data_name + '.pickle', 'rb') as f:
        obj = pickle.load(f)
        f.close()
    data_detection_tr = obj['data_detection_tr']
    labels_detection_tr = obj['labels_detection_tr']
    data_detection_te = obj['data_detection_te']
    labels_detection_te = obj['labels_detection_te']
    data_stream_tr = obj['data_stream_tr']
    labels_stream_tr = obj['labels_stream_tr']
    data_stream_te = obj['data_stream_te']
    labels_stream_te = obj['labels_stream_te']

num_pois = np.sum(data_detection_te == -1)
num_benign = np.sum(data_detection_te == 1)
# train SVM detection model
svm_det = svm.SVC(kernel='linear').fit(data_detection_tr, labels_detection_tr)
predictions = svm_det.predict(data_detection_te)
det_fp = 0
det_fn = 0
for i in range(len(data_detection_te)):
    if predictions[i] == 1 and labels_detection_te[i] == -1:
        det_fn += 1
    elif predictions[i] == -1 and labels_detection_te[i] == 1:
        det_fp += 1
print('SVM detection false negative= ', det_fn/num_pois, '; false positive= ', det_fp/num_benign)

data = data_stream_tr[0]
labels = labels_stream_tr[0]

# RONI filtering
svc_roni = svm.LinearSVC(loss='hinge').fit(data, labels)
err = 1 - svc_roni.score(data, labels)
roni_fp = 0
roni_fn = 0
for data_batch, labels_batch, i in zip(data_stream_te, labels_stream_te, range(len(labels_stream_te))):
    train_set_tmp = np.append(data, data_batch, axis=0)
    train_labels_tmp = np.append(labels, labels_batch)
    svc_roni.fit(train_set_tmp, train_labels_tmp)
    new_err = 1 - svc_roni.score(data, labels)
    if new_err > err and labels_detection_te[i] == 1:
        roni_fp += 1
    if new_err <= err and labels_detection_te[i] == -1:
        roni_fn += 1
print('RONI detection false negative= ', roni_fn/num_pois, '; false positive= ', roni_fp/num_benign)

# SD filtering
kappa = 0.66667
p = 50
n = len(labels)
# construct P
directions = []
for i in range(p):
    # take direction between 2 random points in the training set
    indices = np.random.randint(low=0, high=n, size=2)
    new_dir = data[indices[0]] - data[indices[1]]
    norm_ = np.linalg.norm(new_dir)
    if norm_ > 1e-5:
        new_dir /= norm_
    directions.append(new_dir)

directions = np.array(directions)
data_by_class = {1: np.array([data[i, :] for i in range(n) if labels[i] == 1]),
                 -1: np.array([data[i, :] for i in range(n) if labels[i] == -1])}
outl_test = np.zeros_like(labels_stream_te)
for data_batch, labels_batch, i in zip(data_stream_te, labels_stream_te, range(len(labels_stream_te))):
    # calculate outl for each point
    for j in range(len(labels_batch)):
        outl_meas = 0
        for a in directions:
            sd = abs(a @ data_batch[j] - np.median(data_by_class[labels_batch[j]] @ a)) / robust.scale.mad(
                data_by_class[labels_batch[j]] @ a)
            if sd > outl_meas:
                outl_meas = sd
        outl_test[i] += outl_meas
pred_pois = outl_test.argsort()[int(kappa*n):]

sd_fp = 0
sd_fn = 0
for i in range(len(labels_detection_tr)):
    if labels_detection_te[i] == 1 and i in pred_pois:
        sd_fp += 1
    elif labels_detection_te[i] == -1 and not (i in pred_pois):
        sd_fn += 1

print('SD detection false negative= ', sd_fn/num_pois, '; false positive= ', sd_fp/num_benign)

# Cramer filtering





