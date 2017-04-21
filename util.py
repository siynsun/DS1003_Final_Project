import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error



# divide the dataset to train dataset and test data set
def prepare_train_test_set(datapath, label="Y", test_ratio=0.3):

    #read data
    pkl_file = open(datapath, 'rb')
    dataset = pickle.load(pkl_file)
    dataset = pd.DataFrame(dataset)

    # parition the dataset randomly
    drop_cols = [label]
    x = dataset.drop(drop_cols, 1).as_matrix()
    y = dataset[label].as_matrix()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio)
    return [x_train, x_test, y_train, y_test]


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


