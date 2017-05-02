# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:52:09 2017

@author: siyang
"""

import sys
import os

# Change your path
os.chdir('/Users/siyang/Downloads/DS1003_Final_Project-master/')
sys.path.append('/Users/siyang/Downloads/DS1003_Final_Project-master/')

# Codes starts here

# Import the necessary modules and libraries
#from importlib import reload
#reload(util)
import util
from util import regression_loss
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pickle

# read the data
x_cv, x_test, y_cv, y_test  = util.prepare_train_test_set('./data/df.p')

# choose model

clf = Pipeline([('clf',RandomForestRegressor(criterion='mse',random_state=0))])
#clf = RandomForestRegressor(criterion='mse',random_state=0)

# grid search for the best fit parameters

parameters = {
        'clf__n_estimators': (50, 40, 30, 20, 10), #number of trees
        'clf__max_depth': (75, 50, 40, 30, 25, 10),
        'clf__min_samples_split': (2, 3, 4, 5),
        'clf__min_samples_leaf': (2, 3, 4, 5),
        'clf__min_impurity_split': (1e-8, 1e-7, 1e-6, 1e-5)
    }
    
CV_clf = GridSearchCV(clf, parameters,cv =2,scoring='neg_mean_squared_error')
# Fit regression model
CV_clf.fit(x_cv, y_cv)
CV_result = CV_clf.cv_results_
best_score = np.sqrt(-CV_clf.best_score_)
print ('Best parameters set are: \n %s' %  CV_clf.best_estimator_.get_params())
print('The best RMSE is: %.3f' % best_score)

# save model to pickle
pickle.dump(CV_clf, open("./model/forest_model.pkl", "wb"))

# visualizing purpose
rmse_list = np.sqrt(-CV_result['mean_test_score'])
# Predict

#y_pred = CV_clf.predict(x_test)

# Feature importances

#importances = CV_clf.feature_importances_
#std = np.std([tree.feature_importances_ for tree in CV_clf.estimators_],
             #axis=0)
#indices = np.argsort(importances)[::-1]


# Print the feature ranking
#print("Feature ranking:")

#for f in range(x_train.shape[1]):
    #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the results

#plt.figure()
#plt.title("Feature importances")
#plt.bar(range(x_train.shape[1]), importances[indiches],
 #color="r", yerr=std[indices], align="center")
#plt.xticks(range(x_train.shape[1]), indices)
#plt.xlim([-1, x_train.shape[1]])
#plt.show()


# Visualization 
#import pydot  
#tree.export_graphviz(dtreg, out_file='tree.dot') #produces dot file
#dotfile = StringIO()
#tree.export_graphviz(dtreg, out_file=dotfile)
#pydot.graph_from_dot_data(dotfile.getvalue()).write_png("dtree2.png") 
   
