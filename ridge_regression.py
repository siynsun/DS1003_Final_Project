import sys
import os
os.chdir('/Users/Sean/Desktop/DS1003_Final_Project')
sys.path.append('/Users/Sean/Desktop/DS1003_Final_Project')
from importlib import reload
#reload(util)
import util
from util import regression_loss
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV



# read the data
datapath = './data/encoded_df.pkl'
x_train, x_valid, x_test, y_train, y_valid, y_test = util.prepare_train_test_set(datapath)


# choose model
clf = KernelRidge()

# grid search for the best fit parameters
param_grid = {
    'alpha' : [0.05, 0.1]
}

CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=2)
CV_clf.fit(x_train[1:100,:], y_train[1:100])
print('The best parameters are: \n %s' %CV_clf.best_params_)


# run model and return loss
train_loss, test_loss = util.quick_test_model(x_train[1:100,:], x_test[1:100,:], y_train[1:100], y_test[1:100], CV_clf, regression_loss)

print("Train loss is %s, \n Test loss is %s  " % (train_loss, test_loss))
