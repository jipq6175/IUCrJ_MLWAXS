# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:49:18 2019

@author: Yen-Lin Chen
"""


"""
This is the Python code for training CNN of one helical structural descriptor: helical radius. 
It requires keras 2.2.4 and pickle libraries. 
"""



"""
Part 1: 
Set up the neural network architectures for different datasets.
Specify the hyper-parameters, dataset and optimizer and then get training started. 
"""

# Dependencies: 
import time
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization
from keras.optimizers import Adam
from pickle import dump

# Define the convolutional neural network architecture and construct it.
def convolutional_neural_nets(indim=191, outdim=25, n_filters=128, conv_kernel=10, conv_stride=1, pool_kernel=8, pool_stride=2):

    model = Sequential();

    # first convolutional layers
    model.add(Conv1D(n_filters, kernel_size=conv_kernel, strides=conv_stride, padding='same', input_shape=(indim,1)));
    model.add(Activation('relu'));
    model.add(Dropout(0.2));
    model.add(MaxPooling1D(pool_kernel, pool_stride));

    # Second convolutional layers
    model.add(Conv1D(n_filters, kernel_size=conv_kernel, strides=conv_stride, activation='relu', padding='same'));
    model.add(Dropout(0.2));
    model.add(MaxPooling1D(pool_kernel, pool_stride));

    # Third convolutional layers
    model.add(Conv1D(n_filters, kernel_size=conv_kernel, strides=conv_stride, activation='relu', padding='same'));
    model.add(Dropout(0.2));
    model.add(MaxPooling1D(pool_kernel, pool_stride));

    # Flatten to feed into fully connected nets
    model.add(Flatten());

    # First fully connected nets
    model.add(Dense(units=1024, activation='relu'));
    model.add(BatchNormalization());

    # Second fully connected nets
    model.add(Dense(units=256, activation='relu'));
    model.add(Dropout(0.3));

    # Third fully connected nets
    model.add(Dense(units=64, activation='relu'));
    model.add(Dropout(0.3));

    # To output
    model.add(Dense(units=outdim, activation='softmax'));

    # print the model architecture
    model.summary();

    return model;

# Define the training parameters, train the convolutional neural network and save the model and history
def train_model(model, datadir, x_train_sample, y_train_sample, learning_rate=1e-3, batchsize=32, n_epochs=500):

    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-3, amsgrad=False);
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']);
    hist = model.fit(x_train_sample, y_train_sample, batch_size=batchsize, epochs=n_epochs);
    
    # Evaluate training result
    trainresult = model.evaluate(x_train_sample, y_train_sample, batch_size=batchsize);
    print('Training accuracy = ', trainresult[1]);
    
    # Save the trained model
    model_filename = 'cnn_' + datadir + '_model_dropout_' + time.strftime("%Y%m%d", time.localtime()) + '.h5';
    model.save(model_filename);
    
    # Save the training history
    hist_filename = 'cnn_' + datadir + '_history_dropout_' + time.strftime("%Y%m%d", time.localtime());
    with open(hist_filename, 'wb') as histfile:
        dump(hist.history, histfile);

    return model, hist;


"""
Part 2: 
Prepare the full SWAXS dataset to train the CNN model
"""

# Dependencies
import os
import numpy as np
import h5py as h5

# Load hdf5 files into tensors as training/validation data
def load_helix_data(datadir, datacat='data_train.txt'):
	
	f = open(os.path.join(datadir, datacat));
	training_datalist = f.read();
	training_datalist = training_datalist.split('\n')[0:-1];
	f.close();
	
	# Getting the x_train, y_train
	x_train = np.zeros((1, 191));
	y_train = np.zeros(1);
	for filename in training_datalist:
		h5file = h5.File(os.path.join(datadir, filename));
		x_train = np.vstack((x_train, np.array(h5file['data'][:,:])));
		y_train = np.hstack((y_train, np.array(h5file['label'][:,:])));
		print('INFO: %s Processing Done..' %filename);

    # The first row contains identifiers
	return (x_train[1:, :], y_train[1:, :]);


"""
Part 3: 
Training script
"""

# Set up data directory, for example, helical_radius
datadir = 'F://Yen//DuplexData_Full//helical_radius';

# Load the training data
(x_train, y_train) = load_helix_data(datadir, datacat='data_train.txt');
# Validation data
# (x_validate, y_validate) = load_helix_data(datadir, datacat='data_validate.txt');

# set up the model and train 
model = convolutional_neural_nets(indim=191, outdim=25, n_filters=128, conv_kernel=10, conv_stride=1, pool_kernel=8, pool_stride=2);
model, hist = train_model(model, 'helical_radius', x_train, y_train, learning_rate=1e-4, batchsize=256, n_epochs=2000);


