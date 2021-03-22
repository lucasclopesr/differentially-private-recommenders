import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from matrix_factorization import BaselineModel, KernelMF
from tqdm import tqdm
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

tunne_hparams = False

df = pd.read_csv("../data/slice_ratings_Electronics.csv")
cols = ['user_id','item_id', 'rating','timestamp','Counts Users', 'Counts Items']
df.columns = cols
df.drop(['Counts Users','Counts Items'], axis=1,inplace=True )

X = df[['user_id','item_id']]
y = df['rating']

param_dist = {'n_epochs': [5,10,20,50,100],
              'n_factors': [20,50,100,200,500],
              'lr': list(np.arange(0.001, 0.02, step=0.001)),
              'reg': list(np.arange(0.005, 0.02, step=0.001)),
             }

if tunne_hparams:
    best_rmse = 10000
    hparam = {}
    kfold = KFold(n_splits=5)
    for h in tqdm(model_selection.ParameterSampler(param_dist, n_iter=50, random_state=123456)):
        matrix_fact = KernelMF(n_epochs=h['n_epochs'], n_factors=h['n_factors'], verbose=0, lr=h['lr'], reg=h['reg'])
        rmse = 0
        for train_idx, test_idx in kfold.split(X):
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]

            matrix_fact.fit(X_train, y_train)

            pred = matrix_fact.predict(X_test)
            rmse += mean_squared_error(y_test, pred, squared=False)
        rmse = rmse/5
        print(f"\nTest RMSE: {rmse:.4f}")
        if rmse < best_rmse:
            hparam = h
            best_rmse = rmse
else:
    hparam = {'reg': 0.009000000000000001, 'n_factors': 20, 'n_epochs': 5, 'lr': 0.017}

rmse_cases = {}

kfold = KFold(n_splits=10)
count = 0
rmse = 0
for train_idx, test_idx in kfold.split(X):
    count +=1
    X_train = X.loc[train_idx].copy()
    y_train = y.loc[train_idx].copy()
    X_test = X.loc[test_idx].copy()
    y_test = y.loc[test_idx].copy()

    # Initial training
    matrix_fact = KernelMF(n_epochs=hparam['n_epochs'], n_factors=hparam['n_factors'], verbose=0, lr=hparam['lr'], reg=hparam['reg'])
    matrix_fact.fit(X_train, y_train)

    pred = matrix_fact.predict(X_test)
    rmse += mean_squared_error(y_test, pred, squared=False)
    X_test['rating'] = y_test
    X_test['prediction'] = pred
    X_test.to_csv(f"tests/MF_test_{count}.csv",encoding='utf-8', index=False)
rmse_cases['standard'] = rmse/10

privacies = [3,10,100,1000]
template = 'private_slice_ratings_'

for privacy in privacies:
    file = f"../data/{template}{privacy}.csv"
    # print(f"On file: {file}:\n")
    df = pd.read_csv(file, index_col=0)
    cols = ['user_id','item_id', 'old_rating','timestamp','Counts Users', 'Counts Items', 'rating']
    df.columns = cols
    df.drop(['Counts Users','Counts Items'], axis=1,inplace=True )
    
    X = df[['user_id','item_id']]
    Y = df['rating']
    y = df['old_rating']

    kfold = KFold(n_splits=10)
    count = 0
    rmse = 0
    for train_idx, test_idx in kfold.split(X):
        count +=1
        X_train = X.loc[train_idx].copy()
        y_train = Y.loc[train_idx].copy()
        X_test = X.loc[test_idx].copy()
        y_test = y.loc[test_idx].copy()

        # Initial training
        matrix_fact = KernelMF(n_epochs=hparam['n_epochs'], n_factors=hparam['n_factors'], verbose=0, lr=hparam['lr'], reg=hparam['reg'])
        matrix_fact.fit(X_train, y_train)

        pred = matrix_fact.predict(X_test)
        rmse += mean_squared_error(y_test, pred, squared=False)
        X_test['rating'] = y_test
        X_test['prediction'] = pred
        X_test.to_csv(f"tests/MF_P{privacy}_test_{count}.csv",encoding='utf-8', index=False)
    rmse_cases[f"P{privacy}"] = rmse/10
print(rmse_cases)
