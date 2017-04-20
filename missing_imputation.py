import numpy as np
import pandas as pd
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



