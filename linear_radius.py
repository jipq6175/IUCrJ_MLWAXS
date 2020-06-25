# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 17:35:21 2019

@author: Yen-Lin
"""



# The file containing some useful functions 

import os
import sys
sys.path.append('G:/My Drive/14. CNNWAXS/data');
from prepare_data import *
import xgboost
import pickle
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from matplotlib import pyplot as plt




# Set up data directory
datadir = 'G:/My Drive/14. CNNWAXS/data/helical_radius'
prefix = 'radius_MDsize'
os.chdir(datadir)
file = open(prefix + '_stats.dat', 'w');

# load the training data (training/validation)
(x, y) = load_helix_training_data(datadir)
x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=0.2)

# split the data
# x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=0.2, random_state=42)
(x_test, y_test) = load_helix_test_data(datadir)



# Train the unregularized linear model
linreg = LinearRegression() 
linreg.fit(x_train, y_train)

lin_train_preds = linreg.predict(x_train)
lin_validate_preds = linreg.predict(x_validate)
lin_test_preds = linreg.predict(x_test)


lin_train_mse = metrics.mean_squared_error(y_train, lin_train_preds)
lin_validate_mse = metrics.mean_squared_error(y_validate, lin_validate_preds)
lin_test_mse = metrics.mean_squared_error(y_test, lin_test_preds)

print('---- Linear Regression: ')
print('Training Result: MSE = %f \n' % lin_train_mse)
print('Validation Result: MSE = %f \n' % lin_validate_mse)
print('Testing Result: MSE = %f \n' % lin_test_mse)




# Train Ridge Regression
ridgereg = Ridge(alpha=0.05, normalize=True)
ridgereg.fit(x_train, y_train)

ridge_train_preds = ridgereg.predict(x_train)
ridge_validate_preds = ridgereg.predict(x_validate)
ridge_test_preds = ridgereg.predict(x_test)


ridge_train_mse = metrics.mean_squared_error(y_train, ridge_train_preds)
ridge_validate_mse = metrics.mean_squared_error(y_validate, ridge_validate_preds)
ridge_test_mse = metrics.mean_squared_error(y_test, ridge_test_preds)

print('---- Ridge Regression (0.05): ')
print('Training Result: MSE = %f \n' % ridge_train_mse)
print('Validation Result: MSE = %f \n' % ridge_validate_mse)
print('Testing Result: MSE = %f \n' % ridge_test_mse)



# LASSO
lassoreg = Lasso(alpha=0.2, normalize=True)
lassoreg.fit(x_train, y_train)

lasso_train_preds = lassoreg.predict(x_train)
lasso_validate_preds = lassoreg.predict(x_validate)
lasso_test_preds = lassoreg.predict(x_test)


lasso_train_mse = metrics.mean_squared_error(y_train, lasso_train_preds)
lasso_validate_mse = metrics.mean_squared_error(y_validate, lasso_validate_preds)
lasso_test_mse = metrics.mean_squared_error(y_test, lasso_test_preds)

print('---- LASSO Regression (0.2): ')
print('Training Result: MSE = %f \n' % lasso_train_mse)
print('Validation Result: MSE = %f \n' % lasso_validate_mse)
print('Testing Result: MSE = %f \n' % lasso_test_mse)