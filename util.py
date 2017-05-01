import pickle
import operator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error

# combine two pickle file
def dataset_combine(datapath1, datapath2):
    pkl_file = open(datapath1, 'rb')
    dataset1 = pickle.load(pkl_file)
    dataset1 = pd.DataFrame(dataset1)
    dataset1['room_type_Private room'] = 0
    dataset1['room_type_Entire room'] = 1

    pkl_file = open(datapath2, 'rb')
    dataset2 = pickle.load(pkl_file)
    dataset2 = pd.DataFrame(dataset2)
    dataset2['room_type_Entire room'] = 0

    big_data = dataset1.append(dataset2, ignore_index=True)
    with open('./data/df.p', 'wb') as handle:
        pickle.dump(big_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# divide the dataset to train dataset and test data set
def prepare_train_test_set(datapath, label="price", test_ratio=0.2):
    # read data
    pkl_file = open(datapath, 'rb')
    dataset = pickle.load(pkl_file)
    dataset = pd.DataFrame(dataset)

    # parition the dataset randomly
    x = dataset.drop([label], 1)
    y = dataset[label]
    x_cv, x_test, y_cv, y_test = train_test_split(x, y, test_size=test_ratio, 
                                                    random_state=0)

    # feature importance
    clf = ExtraTreesRegressor(random_state = 0)
    clf = clf.fit(x_cv, y_cv)
    imp = np.array(clf.feature_importances_)
    col = list(x_cv.columns.values)
    keep = []
    imp_record = {}
    for i in range(len(imp)):
        if imp[i] > 0.001:
            imp_record[col[i]] = imp[i]
            keep.append(col[i])
    sorted_imp = sorted(imp_record.items(), key=operator.itemgetter(1))
    print(sorted_imp)
    
    # transform to matrix
    x_cv = x_cv[keep].as_matrix()
    x_test = x_test[keep].as_matrix()
    y_cv = y_cv.as_matrix()
    y_test = y_test.as_matrix()
    
    return [x_cv, x_test, y_cv, y_test]

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