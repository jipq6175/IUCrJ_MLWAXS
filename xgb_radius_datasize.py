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
from matplotlib import pyplot as plt




# Set up data directory
datadir = 'G:/My Drive/14. CNNWAXS/data/helical_radius'
prefix = 'radius_MDsize'
os.chdir(datadir)
file = open(prefix + '_stats.dat', 'w');

# load the training data (training/validation)
(x, y) = load_helix_training_data(datadir)

# split the data
# x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=0.2, random_state=42)
(x_test, y_test) = load_helix_test_data(datadir)




#%
## The noise-free regression model
file.write('# Noise-free model : ')
regressmodel = xgboost.XGBRegressor(colsample_bytree=0.4,
                                    gamma=0,                 
                                    learning_rate=0.07,
                                    max_depth=3,
                                    min_child_weight=1.5,
                                    n_estimators=750,                                                                    
                                    reg_alpha=0.75,
                                    reg_lambda=0.45,
                                    subsample=0.8,
                                    seed=42)



splittings = 1.2*0.8**np.arange(20,0,-1)
cres = []
tres = []
vres = []
test = []



for splitting in splittings:
    
    print('# Splitting = %f \n' % splitting)
    file.write('# Splitting = %f \n' % splitting)
    
    x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=(1.0 - splitting))

    # 10 fold cross validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    cvrlt = cross_val_score(regressmodel, x_train, y_train, scoring='neg_mean_squared_error', cv=kfold, verbose=True)
    print('Noise-free 10-fold Cross Validation Result: MSE= %f (%f) \n' % (-cvrlt.mean(), cvrlt.std()))
    file.write('Noise-free 10-fold Cross Validation Result: MSE= %f (%f) \n' % (-cvrlt.mean(), cvrlt.std()))
    cres.append(-cvrlt.mean())
    
    # training and using early stopping to get our model 
    regressmodel.n_estimators = 7500
    regressmodel.fit(x_train, y_train, 
                     eval_metric='rmse', 
                     eval_set=[(x_validate, y_validate)], 
                     early_stopping_rounds=int(0.01 * regressmodel.n_estimators), 
                     verbose=True)
    
    # See the performance on the training set
    train_preds = regressmodel.predict(x_train)
    train_mse = metrics.mean_squared_error(y_train, train_preds)
    print('Noise-free Training Result: MSE = %f \n' % train_mse)
    file.write('Noise-free Training Result: MSE = %f \n' % train_mse)
    tres.append(train_mse)
    
    # See the performance on the validation set
    validate_preds = regressmodel.predict(x_validate)
    validate_mse = metrics.mean_squared_error(y_validate, validate_preds)
    print('Noise-free Validation Result: MSE = %f \n' % validate_mse) 
    file.write('Noise-free Validation Result: MSE = %f \n' % validate_mse)
    vres.append(validate_mse)
    
    # See the performance on the testing set
    test_preds = regressmodel.predict(x_test)
    test_mse = metrics.mean_squared_error(y_test, test_preds)
    print('Noise-free Testing Result: MSE = %f \n' % test_mse) 
    file.write('Noise-free Testing Result: MSE = %f \n\n' % test_mse)
    test.append(test_mse)


# Save the xgboost models for later 
# print('Saving the noise-free XGBoost model ... ')
# pickle.dump(regressmodel, open(prefix + '_noisefree.xgb.dat', 'wb'))

file.write('End of File')
file.close()


#%%
s = splittings*21542
import matplotlib.gridspec as gridspec

# Set up the plot parameters
from matplotlib import rcParams
DPI = 600;
plotdir = "F:\\Yen\\DuplexData_Full\\FinalPlots"
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['legend.title_fontsize'] = 16
rcParams['font.family'] = 'Times New Roman'

p = plt.figure('different size')
plt.semilogy(s, cres, lw=3.0, alpha=0.8)
plt.semilogy(s, tres, lw=3.0, alpha=0.8)
plt.semilogy(s, vres, lw=3.0, alpha=0.8)
plt.semilogy(s, test, lw=3.0, alpha=0.8)
plt.legend(['Cross Validation', 'Training', 'Validation', 'Testing'])
plt.xlabel('Number of MD structures for Training')
plt.ylabel('MSE')
p.savefig('MDsize.pdf', format='pdf', dpi=DPI);
plt.show();
