import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score, roc_auc_score, \
    confusion_matrix, roc_curve, auc, RocCurveDisplay


def performance_calculation(y_test, y_pred):

    # 1: Autism
    fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)

    y_pred_categorical = np.where(y_pred == 1, 'Autism', 'Control')
    y_test_categorical = np.where(y_test == 1, 'Autism', 'Control')

    cm = confusion_matrix(y_test_categorical, y_pred_categorical, labels=["Autism", "Control"])
    tp = cm[0, 0]
    tn = cm[1, 1]
    fp = cm[1, 0]
    fn = cm[0, 1]

    sen = tp / (tp + fn) if (tp + fn) != 0 else 0
    spc = tn / (tn + fp) if (tn + fp) != 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)
    pre = tp / (tp + fp) if (tp + fp) != 0 else 0  # Also known as PPV
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0
    f1_score_value = 2 * (pre * sen) / (pre + sen) if (pre + sen) != 0 else 0
    roc_auc = np.round(auc(fpr, tpr), 3)

    return tp, tn, fp, fn, sen, spc, acc, pre, npv, f1_score_value, roc_auc
