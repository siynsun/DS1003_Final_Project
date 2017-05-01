import sys
import os
#os.chdir('/Users/Sean/Desktop/DS1003_Final_Project')
#sys.path.append('/Users/Sean/Desktop/DS1003_Final_Project')
#from importlib import reload
#reload(util)
import xgboost as xgb
import util
from util import regression_loss
from sklearn.model_selection import GridSearchCV
import pickle



# read the data
datapath = './data/encoded_others.pkl'
x_train, x_test, y_train, y_test = util.prepare_train_test_set(datapath)


# choose model
clf = xgb.XGBRegressor()

# grid search for the best fit parameters
param_grid = {
    'max_depth': [2, 4, 6],
    'n_estimators': [50, 100, 200]
}

CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=2)
CV_clf.fit(x_train[1:100,:], y_train[1:100])


# save model to pickle
pickle.dump(CV_clf, open("./model/xgboost_model.pkl", "wb"))

print('The best parameters are: \n %s' %CV_clf.best_params_)


# run model and return loss
train_loss, test_loss = util.quick_test_model(x_train[1:100,:], x_test[1:100,:], y_train[1:100], y_test[1:100], CV_clf, regression_loss)

print("Train loss is %s, \n Test loss is %s  " % (train_loss, test_loss))


# load model from file
# model = pickle.load(open("./model/xgboost_model.pkl", "rb"))

