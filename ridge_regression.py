import util
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV

# read the data
datapath = './data/df.p'
x_cv, x_test, y_cv, y_test = util.prepare_train_test_set(datapath)

# exponential
#y_cv = np.exp(y_cv)
#y_test = np.exp(y_test)

# choose model
clf = Ridge(max_iter=3000)

# parameters
param_grid = {
    'alpha': np.logspace(-2, 4, num=13)
}

# grid search
CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=2, 
                      scoring='neg_mean_squared_error')
CV_clf.fit(x_cv, y_cv)
CV_result = CV_clf.cv_results_
best_score = np.sqrt(-CV_clf.best_score_)
print('The best parameters are: %s' %CV_clf.best_params_)
print('The best RMSE is: %.3f' %best_score)

# visualizing purpose
rmse_list = np.sqrt(-CV_result['mean_test_score'])