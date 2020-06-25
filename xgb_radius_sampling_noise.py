# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 17:35:21 2019

@author: Yen-Lin
"""



# The file containing some useful functions 

import os
import sys
sys.path.append('G:/My Drive/14. CNNWAXS/data')
from prepare_data import *
import xgboost
import pickle
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import metrics
from matplotlib import pyplot as plt



# Set up data directory
datadir = 'G:/My Drive/14. CNNWAXS/data/helical_radius'
prefix = 'radius'
os.chdir(datadir)

# load the training data (training/validation)
(x, y) = load_helix_training_data(datadir)
(x_test, y_test) = load_helix_test_data(datadir)

# split the data
x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=0.2, random_state=42)




## Look at the limit of sampling
## Reduce the sampling and train on the same model and use the follow column
file = open(prefix + '_stats_nsample.dat', 'w')
# sample    10-fold CV  Training    Validation  Testing 
file.write('--- The limit of sampling: \n')
print('--- The limit of sampling: \n')
file.write('sample # \t 10-fold CV \t Training MSE \t Validation MSE \t Testing MSE \n')


## from 90 to 10
for nsample in range(95,3,-8):

    file.write('%d\t' % nsample)
    print('--- nsample = %d' % nsample)

    x_train_down = q_interp(x_train, nsample)
    x_validate_down = q_interp(x_validate, nsample)

    downregressmodel = xgboost.XGBRegressor(colsample_bytree=0.4,
                                        gamma=0,                 
                                        learning_rate=0.07,
                                        max_depth=3,
                                        min_child_weight=1.5,
                                        n_estimators=750,                                                                    
                                        reg_alpha=0.75,
                                        reg_lambda=0.45,
                                        subsample=0.8,
                                        seed=42)

    # 10 fold cross validation too
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    cvrlt = cross_val_score(downregressmodel, x_train_down, y_train, scoring='neg_mean_squared_error', cv=kfold, verbose=True)
    print('Down Sampling 10-fold Cross Validation Result: MSE= %f (%f) \n' % (-cvrlt.mean(), cvrlt.std()))
    file.write('%f (%f) \t' % (-cvrlt.mean(), cvrlt.std()))

    downregressmodel.n_estimators = 7500
    downregressmodel.fit(x_train_down, y_train, 
                    eval_metric='rmse', 
                    eval_set=[(x_validate_down, y_validate)], 
                    early_stopping_rounds=int(0.01 * downregressmodel.n_estimators), 
                    verbose=True)

    # See the performance on the training set
    train_preds = downregressmodel.predict(x_train_down)
    train_mse = metrics.mean_squared_error(y_train, train_preds)
    print('Down Sampling Training Result: MSE = %f \n' % train_mse)
    file.write('%f \t' % train_mse)

    # See the performance on the validation set
    validate_preds = downregressmodel.predict(x_validate_down)
    validate_mse = metrics.mean_squared_error(y_validate, validate_preds)
    print('Down Sampling Validation Result: MSE = %f \n' % validate_mse) 
    file.write('%f \t' % validate_mse)

    # See the performance on the testing set
    x_test_down = q_interp(x_test, nsample)
    test_preds = downregressmodel.predict(x_test_down)
    test_mse = metrics.mean_squared_error(y_test, test_preds)
    print('Down Sampling Testing Result: MSE = %f \n' % test_mse) 
    file.write('%f \n' % test_mse)

    # Save the xgboost models for later 
    print('Saving the Down Sampling XGBoost model ... ')
    pickle.dump(downregressmodel, open(prefix + '_sam%d.xgb.dat' % nsample, 'wb'))

print('--- The limit of sampling: ALL DONE!!\n')
file.write('End of File')
file.close()










## Look at the limit of noise
## Increase the noise and train on the same model and use the follow column
file = open(prefix + '_stats_noise.dat', 'w')
# noise    10-fold CV  Training    Validation  Testing 
file.write('--- The limit of noise: \n')
print('--- The limit of noise: \n')
file.write('noise \t 10-fold CV \t Training MSE \t Validation MSE \t Testing MSE \n')

for noise in np.linspace(0.02, 0.25, 24):
    
    file.write('%f\t' % noise)
    print('--- noise = %f' % noise)


    x_train_noise = np.log10(np.multiply(10**x_train, 1 + noise*np.random.rand(x_train.shape[0], x_train.shape[1])))
    x_validate_noise = np.log10(np.multiply(10**x_validate, 1 + noise*np.random.rand(x_validate.shape[0], x_train.shape[1])))

    noiseregressmodel = xgboost.XGBRegressor(colsample_bytree=0.4,
                                        gamma=0,                 
                                        learning_rate=0.07,
                                        max_depth=3,
                                        min_child_weight=1.5,
                                        n_estimators=750,                                                                    
                                        reg_alpha=0.75,
                                        reg_lambda=0.45,
                                        subsample=0.8,
                                        seed=42)

    # 10 fold cross validation too
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    cvrlt = cross_val_score(noiseregressmodel, x_train_noise, y_train, scoring='neg_mean_squared_error', cv=kfold, verbose=True)
    print('Noisy 10-fold Cross Validation Result: MSE= %f (%f) \n' % (-cvrlt.mean(), cvrlt.std()))
    file.write('%f (%f) \t' % (-cvrlt.mean(), cvrlt.std()))

    noiseregressmodel.n_estimators = 7500
    noiseregressmodel.fit(x_train_noise, y_train, 
                    eval_metric='rmse', 
                    eval_set=[(x_validate_noise, y_validate)], 
                    early_stopping_rounds=int(0.01 * noiseregressmodel.n_estimators), 
                    verbose=True)

    # See the performance on the training set
    train_preds = noiseregressmodel.predict(x_train_noise)
    train_mse = metrics.mean_squared_error(y_train, train_preds)
    print('Noisy Training Result: MSE = %f \n' % train_mse)
    file.write('%f \t' % train_mse)

    # See the performance on the validation set
    validate_preds = noiseregressmodel.predict(x_validate_noise)
    validate_mse = metrics.mean_squared_error(y_validate, validate_preds)
    print('Noisy Validation Result: MSE = %f \n' % validate_mse) 
    file.write('%f \t' % validate_mse)

    # See the performance on the testing set
    x_test_noise = np.log10(np.multiply(10**x_test, 1 + noise*np.random.rand(x_test.shape[0], x_test.shape[1])))
    test_preds = noiseregressmodel.predict(x_test_noise)
    test_mse = metrics.mean_squared_error(y_test, test_preds)
    print('Noisy Testing Result: MSE = %f \n' % test_mse) 
    file.write('%f \n' % test_mse)

    # Save the xgboost models for later 
    print('Saving the noisy XGBoost model ... ')
    pickle.dump(noiseregressmodel, open(prefix + '_noisy%f.xgb.dat' % noise, 'wb'))


print('--- The limit of noise: ALL DONE!!\n')
file.write('End of File')
file.close()
