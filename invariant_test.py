from sklearn import svm
from sklearn.metrics import accuracy_score
import data as dt
import functions as fn
import numpy as np
from art.attacks import PoisoningAttackSVM
from art.classifiers import SklearnClassifier


n = 100
m = 10
a = -10
b_ = 10
n_bins = 50
n_pois = 50
n_sample = 20
data, labels = dt.get_toy_dataset(n, m, random_flips=0.05)
dist = 0.1*np.linalg.norm(data)*n_pois/len(data)
# train svm
svm_ = svm.SVC(kernel='linear').fit(data, labels)
w = svm_.coef_[0]
b = svm_.intercept_[0]
# get normalized histogram
x_norm = np.array([labels[i]*(np.dot(data[i], w)+b)/np.linalg.norm(w) for i in range(n)])
hist_orig = fn.get_histogram(x_norm, a, b_, n_bins)[0]
print('histogram of original data: ', hist_orig)
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
target_sets = [data_detection_tr, data_detection_te, data_detection_te]
target_labels = [labels_detection_tr, labels_detection_te, labels_detection_te]
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
        pois_x = np.array([(1 if pois_labels[i, 0] == 1 else -1) *
                           (np.dot(pois_data[i], w)+b)/np.linalg.norm(w) for i in range(n_pois)])
        target_sets[j].append(fn.get_histogram(pois_x, a, b_, n_bins)[0])
        target_labels[j].append(-1)

    # generate covariate shifts
    for i in range(n_pois):
        indices = np.random.randint(0, n, n_pois)
        cov_shift_data = data[indices]
        cov_shift_labels = labels[indices]
        cov_shift_x = np.array([cov_shift_labels[i] *
                                (np.dot(cov_shift_data[i], w)+b)/np.linalg.norm(w) for i in range(n_pois)])
        target_sets[j].append(fn.get_histogram(cov_shift_x, a, b_, n_bins)[0])
        target_labels[j].append(1)

    # generate concept drifts
    for i in range(n_pois):
        indices = np.random.randint(0, n, n_pois)
        conc_drift_data, conc_drift_labels = dt.get_random_concept_drift(data[indices], labels[indices], dist)
        conc_drift_x = np.array([conc_drift_labels[i] *
                                (np.dot(conc_drift_data[i], w)+b)/np.linalg.norm(w) for i in range(n_pois)])
        target_sets[j].append(fn.get_histogram(conc_drift_x, a, b_, n_bins)[0])
        target_labels[j].append(1)
    if j == 0:
        print('training data generated!')
print('test data generated!')

# train detection model
data_detection_tr = np.array(data_detection_tr)
labels_detection_tr = np.array(labels_detection_tr)
data_detection_te = np.array(data_detection_te)
labels_detection_te = np.array(labels_detection_te)
svm_det = svm.SVC(kernel='linear').fit(data_detection_tr, labels_detection_tr)
predictions = svm_det.predict(data_detection_te)
err = 1 - accuracy_score(labels_detection_te, predictions)
print('Poisoning detection error: {}%'.format(err * 100))
print('svm normal vector:', svm_det.coef_[0])


