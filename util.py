import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import r2_score, mean_squared_error

# divide the dataset to train dataset and test data set
def prepare_train_test_set(datapath, label="price",
                           valid_ratio=0.2, test_ratio=0.2):
    # read data
    pkl_file = open(datapath, 'rb')
    dataset = pickle.load(pkl_file)
    dataset = pd.DataFrame(dataset)

    # parition the dataset randomly
    x = dataset.drop([label], 1)
    y = dataset[label]
    x_sub, x_test, y_sub, y_test = train_test_split(x, y, test_size=test_ratio, 
                                                    random_state=0)
    x_train, x_valid, y_train, y_valid = train_test_split(x_sub, y_sub, test_size=valid_ratio, 
                                                          random_state=0)
    
    # feature importance
    clf = ExtraTreesClassifier(random_state = 0)
    clf = clf.fit(x_train, y_train)
    imp = np.array(clf.feature_importances_)
    col = list(x_train.columns.values)
    keep = []
    for i in range(len(imp)):
        if imp[i] > 0.001:
            keep.append(col[i])
    
    # transform to matrix
    x_train = x_train[keep].as_matrix()
    x_valid = x_valid[keep].as_matrix()
    x_test = x_test[keep].as_matrix()
    y_train = y_train.as_matrix()
    y_valid = y_valid.as_matrix()
    y_test = y_test.as_matrix()
    
    return [x_train, x_valid, x_test, y_train, y_valid, y_test]


def quick_test_model(x_train, x_test, y_train, y_test, model, eval_metrics):
    #all_labels = np.unique(np.concatenate([y_train, y_test]))
    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    train_loss = eval_metrics(y_train, pred_train)
    test_loss = eval_metrics(y_test, pred_test)
    return [train_loss, test_loss]


# cv tune parameter based on rmse score
def rmse_cv(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=2))
    return rmse

# score function
def regression_loss(prediction, lables):
    loss = [('R2: {}'.format(r2_score(prediction, lables))), ('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))]
    return loss