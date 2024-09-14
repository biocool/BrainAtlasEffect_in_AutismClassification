from scipy.stats import norm
import glob
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer, MinMaxScaler
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
import os
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score, roc_auc_score, \
    confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.feature_selection import VarianceThreshold
import numpy as np
from sklearn.model_selection import StratifiedKFold
from Library.neuroCombat import neuroCombat
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import plotting
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
import random
from scipy.stats import loguniform
import gzip
import pickle
from collections import Counter
from loadData import Load_Atlas_Data
from performanceCalculation import performance_calculation


def generate_clf():
    ada = AdaBoostClassifier(n_estimators=200, estimator=None, learning_rate=0.1, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    nivebase = GaussianNB()
    dt = DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_leaf=5, random_state=42)
    lr = LogisticRegression(random_state=42)
    svclassifier = SVC(kernel='rbf', degree=5, C=2, random_state=42)
    ridge = RidgeClassifier(alpha=0.5, random_state=42)
    sgdclassifier = SGDClassifier(random_state=42)
    randomforest = RandomForestClassifier(n_estimators=42)  # Train the model on training data
    mlp = MLPClassifier(hidden_layer_sizes=(20, 10, 5), max_iter=1000, alpha=1e-4, solver='sgd', verbose=0,
                        tol=1e-4, learning_rate_init=.08, random_state=42)

    bagging = BaggingClassifier(dt, max_samples=0.5, max_features=0.5, random_state=42)

    classifier_dict = {'KNeighborsClassifier': knn, 'GaussianNB': nivebase, 'DecisionTreeClassifier': dt,
                       'LogisticRegression': lr, 'SVC': svclassifier, 'RandomForestClassifier': randomforest,
                       'RidgeClassifier': ridge, 'MLPClassifier': mlp, 'BaggingClassifier': bagging,
                       'SGDClassifier': sgdclassifier, 'AdaBoostClassifier': ada}

    return classifier_dict


def run_cross_validation(X, y, classifier, kf, clf_name, atlas_name):

    Data = X
    itr = 0

    tp_list = []
    tn_list = []
    fp_list = []
    fn_list = []
    sen_list = []
    spc_list = []
    acc_list = []
    pre_list = []
    npv_list = []
    f1_score_value_list = []
    roc_auc_list = []

    for train_index, test_index in kf.split(Data, y):

        classifier.fit(Data[train_index], y[train_index])
        y_pred = classifier.predict(Data[test_index])
        y_test = y[test_index]

        tp, tn, fp, fn, sen, spc, acc, pre, npv, f1_score_value, roc_auc = performance_calculation(y_test, y_pred)

        row = pd.DataFrame(data={'clf': [clf_name], 'atlas_name': [atlas_name], 'itr': [itr], 'tp': [tp], 'tn': [tn],
                                 'fp': [fp], 'fn': [fn], 'sen': [sen], 'spc': [spc], 'acc': [acc], 'pre': [pre],
                                 'npv': [npv], 'f1': [f1_score_value], 'roc_auc': [roc_auc]})

        tp_list.append(tp)
        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)
        sen_list.append(sen)
        spc_list.append(spc)
        acc_list.append(acc)
        pre_list.append(pre)
        npv_list.append(npv)
        f1_score_value_list.append(f1_score_value)
        roc_auc_list.append(roc_auc)

        if itr == 0:
            pref_df = row
        else:
            pref_df = pd.concat([pref_df, row], ignore_index=True)

        itr += 1

    # calculate average
    avg_row = pd.DataFrame(data={'clf': [clf_name], 'atlas_name': [atlas_name], 'itr': ['average'],
                                 'tp': [np.average(tp_list)],
                                 'tn': [np.average(tn_list)], 'fp': [np.average(fp_list)], 'fn': [np.average(fn_list)],
                                 'sen': [np.average(sen_list)], 'spc': [np.average(spc_list)],
                                 'acc': [np.average(acc_list)], 'pre': [np.average(pre_list)],
                                 'npv': [np.average(npv_list)], 'f1': np.average(f1_score_value_list),
                                 'roc_auc': [np.average(roc_auc_list)]})

    pref_df = pd.concat([pref_df, avg_row], ignore_index=True)

    return pref_df


def DoCrossValidation(X, Y, kfold, atlas_name):

    kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
    classifier_names = []

    clf_dict = generate_clf()

    for cnt, classifier_name in enumerate(list(clf_dict.keys())):

        classifier_names.append(classifier_name)
        clf = clf_dict[classifier_name]

        performance_df = run_cross_validation(X, Y, classifier=clf, kf=kf, clf_name=classifier_name,
                                              atlas_name=atlas_name)

        if cnt == 0:
            total_performance_df = performance_df
        else:
            total_performance_df = pd.concat([total_performance_df, performance_df], ignore_index=True)

    return total_performance_df


def non_combat():

    cnt = 0
    for folder in glob.glob(abide_dataset_dir + '/*'):

        print(folder)
        print(os.path.basename(folder))
        print('==============================================================')
        X_abide, Y_abide, batch_data = Load_Atlas_Data(folder, extension='1D', metadata_df=metadata_df, mode='abide')
        abide_total_performance_df = DoCrossValidation(X_abide, Y_abide, kfold=10, atlas_name=os.path.basename(folder))

        if cnt == 0:
            final_performance_df = abide_total_performance_df
        else:
            final_performance_df = pd.concat([final_performance_df, abide_total_performance_df], ignore_index=True)

        cnt += 1

    for folder in glob.glob(nilearn_dataset_dir + '/*'):
        print(folder)
        print(os.path.basename(folder))
        print('==============================================================')
        X_nilearn, Y_nilearn, batch_data = \
            Load_Atlas_Data(folder, extension='npy', metadata_df=metadata_df, mode='nilearn')
        nilearn_total_performance_df = DoCrossValidation(X_nilearn, Y_nilearn, kfold=10,
                                                         atlas_name=os.path.basename(folder))

        final_performance_df = pd.concat([final_performance_df, nilearn_total_performance_df], ignore_index=True)

    final_performance_df.to_csv('non_neuroCombat.csv')


def with_combat():

    cnt = 0
    for folder in glob.glob(abide_dataset_dir + '/*'):

        print(folder)
        print(os.path.basename(folder))
        print('==============================================================')
        X_abide, Y_abide, covars = Load_Atlas_Data(folder, extension='1D', metadata_df=metadata_df, mode='abide')

        # normalization
        categorical_cols = ['gender', 'age']
        batch_col = 'batch'
        X_abide = np.transpose(X_abide)
        data_combat = neuroCombat(dat=X_abide, covars=covars, batch_col=batch_col,
                                  categorical_cols=categorical_cols)["data"]
        normalized_X_abide = np.transpose(data_combat)

        abide_total_performance_df = DoCrossValidation(normalized_X_abide, Y_abide, kfold=10,
                                                       atlas_name=os.path.basename(folder))

        if cnt == 0:
            final_performance_df = abide_total_performance_df
        else:
            final_performance_df = pd.concat([final_performance_df, abide_total_performance_df], ignore_index=True)

        cnt += 1

    for folder in glob.glob(nilearn_dataset_dir + '/*'):
        print(folder)
        print(os.path.basename(folder))
        print('==============================================================')
        X_nilearn, Y_nilearn, covars = Load_Atlas_Data(folder, extension='npy', metadata_df=metadata_df, mode='nilearn')

        # normalization
        X_nilearn = np.transpose(X_nilearn)
        categorical_cols = ['gender', 'age']
        batch_col = 'batch'
        data_combat = neuroCombat(dat=X_nilearn, covars=covars, batch_col=batch_col, categorical_cols=categorical_cols)[
            "data"]
        normalized_X_nilearn = np.transpose(data_combat)

        nilearn_total_performance_df = DoCrossValidation(normalized_X_nilearn, Y_nilearn, kfold=10,
                                                         atlas_name=os.path.basename(folder))

        final_performance_df = pd.concat([final_performance_df, nilearn_total_performance_df], ignore_index=True)

    final_performance_df.to_csv('neuroCombat.csv')


if __name__ == '__main__':

    metadata_df = pd.read_csv('Labels.csv')

    abide_dataset_dir = 'AtlasExtracted/Abide/'
    nilearn_dataset_dir = 'AtlasExtracted/NilearnNew/'

    non_combat()
    with_combat()
