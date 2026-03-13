from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, auc
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_curve
import numpy as np

def scores(y_test, y_pred, th=0.5):
    y_predlabel = [(0 if item < th else 1) for item in y_pred]
    y_test = np.array([(0 if item < 1 else 1) for item in y_test])
    y_predlabel = np.array(y_predlabel)
    tn, fp, fn, tp = confusion_matrix(y_test, y_predlabel).flatten()
    SP = tn * 1.0 / ((tn + fp) * 1.0)
    SN = tp * 1.0 / ((tp + fn) * 1.0)
    MCC = matthews_corrcoef(y_test, y_predlabel)
    Recall = recall_score(y_test, y_predlabel)
    Precision = precision_score(y_test, y_predlabel)
    F1 = f1_score(y_test, y_predlabel)
    Acc = accuracy_score(y_test, y_predlabel)
    AUC = roc_auc_score(y_test, y_pred)
    precision_aupr, recall_aupr, _ = precision_recall_curve(y_test, y_pred)
    AUPR = auc(recall_aupr, precision_aupr)
    return Recall, SN, SP, MCC, Precision, F1, Acc, AUC, AUPR, tp, fn, tn, fp

def Aiming(y_hat, y):
    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y_hat[v])
    return sorce_k / n

def Coverage(y_hat, y):
    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y[v])
    return sorce_k / n

def Accuracy(y_hat, y):
    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / union
    return sorce_k / n

def AbsoluteTrue(y_hat, y):
    n, m = y_hat.shape
    score_k = 0
    for v in range(n):
        if list(y_hat[v]) == list(y[v]):
            score_k += 1
    return score_k / n

def AbsoluteFalse(y_hat, y):
    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        sorce_k += (union - intersection) / m
    return sorce_k / n

def evaluate_metrics(y_hat, y):
    score_label = y_hat.copy()
    aiming_list = []
    coverage_list = []
    accuracy_list = []
    absolute_true_list = []
    absolute_false_list = []

    # 转换预测标签为二值标签
    for i in range(len(score_label)):
        for j in range(len(score_label[i])):
            if score_label[i][j] < 0.5:
                score_label[i][j] = 0
            else:
                score_label[i][j] = 1

    y_hat = score_label

    aiming = Aiming(y_hat, y)
    aiming_list.append(aiming)
    coverage = Coverage(y_hat, y)
    coverage_list.append(coverage)
    accuracy = Accuracy(y_hat, y)
    accuracy_list.append(accuracy)
    absolute_true = AbsoluteTrue(y_hat, y)
    absolute_true_list.append(absolute_true)
    absolute_false = AbsoluteFalse(y_hat, y)
    absolute_false_list.append(absolute_false)
    return dict(aiming=aiming, coverage=coverage, accuracy=accuracy, absolute_true=absolute_true,
                absolute_false=absolute_false)