import tensorflow as tf
from statsmodels import robust
from sklearn.metrics import accuracy_score
import numpy as np
import sklearn.svm as svm


def sd_val_success(images, labels):
    kappa = 0.98
    p = 50
    n = len(images)
    # construct P
    directions = []
    for i in range(p):
        # take direction between 2 random points in the training set
        indices = np.random.randint(low=0, high=n, size=2)
        new_dir = images[indices[0]] - images[indices[1]]
        norm_ = np.linalg.norm(new_dir)
        if norm_ > 1e-5:
            new_dir /= norm_
        directions.append(new_dir)

    directions = np.array(directions)

    # separate training set
    train_dataset_7 = []
    indices_7 = []
    train_dataset_9 = []
    indices_9 = []

    for i in range(n):
        if labels[i, 0] == 1.:
            train_dataset_7.append(images[i])
            indices_7.append(i)
        else:
            train_dataset_9.append(images[i])
            indices_9.append(i)

    train_dataset_7 = np.array(train_dataset_7)
    n_7 = len(train_dataset_7)
    n_7_refined = int(np.floor(n_7 * kappa))
    train_dataset_9 = np.array(train_dataset_9)
    n_9 = len(train_dataset_9)
    n_9_refined = int(np.floor(n_9 * kappa))

    # calculate SD outlyingness for 7
    sd_7 = np.zeros(n_7)
    for i in range(n_7):
        for a in directions:
            sd = abs(a @ train_dataset_7[i] - np.median(train_dataset_7 @ a)) / robust.scale.mad(
                train_dataset_7 @ a)
            if sd > sd_7[i]:
                sd_7[i] = sd

    # calculate SD outlyingness for 9
    sd_9 = np.zeros(n_9)
    for i in range(n_9):
        for a in directions:
            sd = abs(a @ train_dataset_9[i] - np.median(train_dataset_9 @ a)) / robust.scale.mad(
                train_dataset_9 @ a)
            if sd > sd_9[i]:
                sd_9[i] = sd

    indices_refined_7 = np.array(indices_7)[sd_7.argsort()[:n_7_refined]]
    indices_refined_9 = np.array(indices_9)[sd_9.argsort()[:n_9_refined]]

    validation_success = []
    for i in range(n):
        validation_success.append((i in indices_refined_7) or (i in indices_refined_9))

    return validation_success


part_stat_7 = 0
part_stat_9 = 0
valid_7_ind = []
valid_9_ind = []


def cramer_test(images, labels, valid_set, valid_indices):
    crit_val = 30.5
    global part_stat_7, part_stat_9, valid_7_ind, valid_9_ind
    if part_stat_7 == 0:
        for i in range(len(valid_indices)):
            if valid_indices[i, 0] == 1:
                valid_7_ind.append(i)
            else:
                valid_9_ind.append(i)
        for i in valid_7_ind:
            for j in valid_7_ind:
                part_stat_7 += np.linalg.norm(valid_set[i] - valid_set[j])
        part_stat_7 /= 2 * (len(valid_7_ind) ** 2)
        if len(valid_9_ind) != 0:
            for i in valid_9_ind:
                for j in valid_9_ind:
                    part_stat_9 += np.linalg.norm(valid_set[i] - valid_set[j])
            part_stat_9 /= 2 * (len(valid_9_ind) ** 2)

    validation_succes = []
    for k in range(len(labels)):
        if labels[k, 0] == 1:
            # 7
            test_stat_7 = part_stat_7
            for j in valid_7_ind:
                test_stat_7 += np.linalg.norm(images[k] - valid_set[j]) / (len(valid_7_ind))
            test_stat_7 *= len(valid_7_ind) / (1 + len(valid_7_ind))
            validation_succes.append(test_stat_7 < crit_val)
        else:
            # 9
            test_stat_9 = part_stat_9
            for j in valid_9_ind:
                test_stat_9 += np.linalg.norm(images[k] - valid_set[j]) / (len(valid_9_ind))
            test_stat_9 *= len(valid_9_ind) / (1 + len(valid_9_ind))
            validation_succes.append(test_stat_9 < crit_val)

    return validation_succes