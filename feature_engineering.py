#to do: missing indicator, mvp
from sklearn import preprocessing
from scipy import stats
import numpy as np

# check skewness for all numerical variables(including price) and log transform skewed numeric features
def log_skewness(data, num_cols, Y):
    #numeric_feats = data.dtypes[data.dtypes != "object"].index
    #skewed_feats = data[num_cols].apply(lambda x: stats.skew(x))  # compute skewness
    #skewed_feats = skewed_feats[skewed_feats > 0.8]
    #skewed_feats = skewed_feats.index
    #data[skewed_feats] = np.log1p(data[skewed_feats])

    data[Y] = np.log(data[Y]) # log(price)

    return data


# detect outliers of target price and delete outliers
# outliers are price which outside 2 standard deviations(or 1.96) # 2.575
def del_outliers(data, Y='price'):
    outliers = []
    data_array = np.array(data[Y])
    mean = np.mean(data_array)
    std = np.std(data_array)
    for i in range(0, len(data_array)):
        if abs(data_array[i] - mean) > 1.96 * std:
            outliers.append(data_array[i])
    thred = sorted(outliers)[0]
    data[Y][data[Y] > thred] = np.nan
    data = data.dropna()
    return data


# standardization
#data_scaled = preprocessing.scale(data)

def standardize_matrix(matrix):
    for col in range(matrix.shape[1]):
        mean = np.mean(matrix[:,col])
        std = np.std(matrix[:,col])
        matrix[:, col] = list(map(lambda x: (x - mean) / std, matrix[:,col]))
    return matrix


def standardize_df(data):
    for col in data.columns.tolist():
        if col != 'price': # not standardize target
            mean = np.mean(data[col])
            std = np.std(data[col] )
            data[col] = data[col].apply(lambda x: (x - mean) / std)
    return data



#pkl_file = open(datapath, 'rb')
#dataset = pickle.load(pkl_file)

