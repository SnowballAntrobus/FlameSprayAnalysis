import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneOut, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def translate(arr):
    s_arr = []
    if arr[0] == 1:
        s_arr.append('LFR')
    elif arr[1] == 1:
        s_arr.append('AO2')
    elif arr[2] == 1:
        s_arr.append('SO2')
    elif arr[3] == 1:
        s_arr.append('ER1')
    elif arr[4] == 1:
        s_arr.append('ER2')
    elif arr[5] == 1:
        s_arr.append('BCT')
    elif arr[6] == 1:
        s_arr.append('LFR*LFR')
    elif arr[7] == 1:
        s_arr.append('LFR*AO2')
    elif arr[8] == 1:
        s_arr.append('LFR*SO2')
    elif arr[9] == 1:
        s_arr.append('LFR*ER1')
    elif arr[10] == 1:
        s_arr.append('LFR*ER2')
    elif arr[11] == 1:
        s_arr.append('LFR*BCT')
    elif arr[12] == 1:
        s_arr.append('AO2*SO2')
    elif arr[13] == 1:
        s_arr.append('AO2*ER1')
    elif arr[14] == 1:
        s_arr.append('AO2*ER2')
    elif arr[15] == 1:
        s_arr.append('AO2*BCT')
    elif arr[16] == 1:
        s_arr.append('SO2*SO2')
    elif arr[17] == 1:
        s_arr.append('SO2*ER1')
    elif arr[18] == 1:
        s_arr.append('SO2*ER2')
    elif arr[19] == 1:
        s_arr.append('SO2*BCT')
    elif arr[20] == 1:
        s_arr.append('ER1*ER1')
    elif arr[21] == 1:
        s_arr.append('ER1*ER2')
    elif arr[22] == 1:
        s_arr.append('ER1*BCT')
    elif arr[23] == 1:
        s_arr.append('ER2*ER2')
    elif arr[24] == 1:
        s_arr.append('ER2*BCT')
    elif arr[25] == 1:
        s_arr.append('BCT*BCT')
    return s_arr


def col_rand():
    s = ''
    while True:
        r = random.randint(0, 6)
        if r == 0:
            s += 'LFR'
        elif r == 1:
            s += 'AO2'
        elif r == 2:
            s += 'SO2'
        elif r == 3:
            s += 'ER1'
        elif r == 4:
            s += 'ER2'
        elif r == 5:
            s += 'BCT'
        else:
            break
        r = random.randint(0, 10)
        if r > 8:
            s += '*'
        else:
            break
    return s


def modify(input_file, columns):
    df = pd.read_csv(input_file, header=0)
    dataset = df.values

    LFR = dataset[:, 1]
    AO2 = dataset[:, 2]
    SO2 = dataset[:, 3]
    ER1 = dataset[:, 4]
    ER2 = dataset[:, 5]
    BCT = dataset[:, 6]

    result = dataset[:, 7]
    df_mod = pd.DataFrame(result, columns=["Result"])

    for op in columns:
        if op == '':
            continue
        tok = op.split('*')
        if tok[0] == 'LFR':
            col = LFR
        elif tok[0] == 'AO2':
            col = AO2
        elif tok[0] == 'SO2':
            col = SO2
        elif tok[0] == 'ER1':
            col = ER1
        elif tok[0] == 'ER2':
            col = ER2
        elif tok[0] == 'BCT':
            col = BCT
        for t in tok[1:]:
            if t == 'LFR':
                col *= LFR
            elif t == 'AO2':
                col *= AO2
            elif t == 'SO2':
                col *= SO2
            elif t == 'ER1':
                col *= ER1
            elif t == 'ER2':
                col *= ER2
            elif t == 'BCT':
                col *= BCT
        df_mod[op] = col

    df_mod.to_csv('modified.csv', index=False)


def evaluate(columns):
    modify("data.csv", columns)

    df = pd.read_csv("modified.csv", header=0)

    dataset = df.values
    X = dataset[:, 1:]
    y = dataset[:, 0]
    y = y.astype('int')

    scale = StandardScaler().fit(X)
    X_std = scale.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_std, y, train_size=.9)

    loo = LeaveOneOut()
    loo.get_n_splits(X_train)

    parameters = {'solver': ['newton-cg', 'lbfgs', 'liblinear'], 'C': [1, 100, 1000]}
    log = LogisticRegression(multi_class='auto', max_iter=1000)
    clf = GridSearchCV(log, parameters, cv=loo)
    clf.fit(X_train, y_train)
    a = clf.best_score_

    parameters = {'kernel': ['rbf', 'linear', 'poly'], 'C': [1, 100, 1000]}
    svc = SVC(gamma="scale")
    clf = GridSearchCV(svc, parameters, cv=loo)
    clf.fit(X_train, y_train)
    if clf.best_score_:
        a = clf.best_score_

    parameters = {'n_neighbors': [2, 3, 4, 5, 6], 'p': [1, 2]}
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, parameters, cv=loo)
    clf.fit(X_train, y_train)
    if clf.best_score_:
        a = clf.best_score_

    return a
