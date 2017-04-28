import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor as KNNr
from sklearn.neighbors import KNeighborsClassifier as KNNc
from sklearn import preprocessing



import os
os.chdir('/Users/Sean/Desktop/DS1003_Final_Project')

def mean_imputation(df, num_cols):
    for col in num_cols:
        df[col].fillna(df[col].mean(), inplace=True)


def text_imputation(df, text_cols):
    df[text_cols] = df[text_cols].fillna('')


def mode_imputation(df, cat_cols):
    for col in cat_cols:
        m = df[col]
        most_common = pd.get_dummies(m).sum().sort_values(ascending=False).index[0]
        df[col].fillna(most_common, inplace=True)


def validate_missing(df):
    missing_list = []
    for i in range(len(df.columns)):
        missing_perc = float(df[[i]].isnull().sum() / len(df[[i]]))
        missing = [missing_perc, df.columns[i]]
        missing_list.append(missing)
    return missing_list


def knn_imputation(data, column_name, k, cat_cols, num_cols):

    mydata = data[cat_cols + num_cols]

    if column_name in cat_cols:
        cat_cols_new = cat_cols.copy()
        cat_cols_new.remove(column_name)
    else:
        num_cols_new = num_cols.copy()
        num_cols_new.remove(column_name)

    # save target with missing value
    y = mydata[column_name].copy()
    mydata = mydata.drop(column_name, axis=1)

    # impute missing temporarily
    df = mydata.copy()
    if column_name in cat_cols:
        mean_imputation(df, num_cols)
        mode_imputation(df, cat_cols_new)
    else:
        mean_imputation(df, num_cols_new)
        mode_imputation(df, cat_cols)

    # encode categorical variable temporarily
    if column_name in cat_cols:
        df = pd.get_dummies(df, columns=cat_cols_new)
    else:
        df = pd.get_dummies(df, columns=cat_cols)

    # put target missing variable back
    df[column_name] = y

    df_missing = df[pd.isnull(df).any(axis=1)]
    df_filled = df[~pd.isnull(df).any(axis=1)]
    y_filled = df_filled[column_name].copy()

    #le = preprocessing.LabelEncoder()
    #le.fit(y)
    #y = le.transform(y)

    X_filled = df_filled.drop(column_name, axis=1)
    X_missing = df_missing.drop(column_name, axis=1)

    if df_filled[column_name].dtype == np.dtype('float64'):
        knn = KNNr(n_neighbors=k)
    else:
        knn = KNNc(n_neighbors=k)
    knn.fit(X_filled, y_filled)

    # using knn to predict missing value
    df_missing[column_name] = knn.predict(X_missing)

    # replace prediction to nan at missing target column
    df_final = data.copy()
    df_final[column_name].fillna(df_missing[column_name], inplace=True)

    return df_final


