# script for training the helical_radius

import sys
sys.path.append('F://Yen//DuplexData_Full');
from prepare_data import *
from training import *

# Set up data directory, for example, helical_radius
datadir = 'F://Yen//DuplexData_Full//helical_radius';

# Load the data
(x_train, y_train) = load_helix_data(datadir, datacat='data_train.txt');

# set up the model and train 
# model = convolutional_neural_nets(indim=191, outdim=25, n_filters=128, conv_kernel=10, conv_stride=1, pool_kernel=8, pool_stride=2);
# model, hist = train_model(model, 'helical_radius', x_train, y_train, learning_rate=1e-4, batchsize=256, n_epochs=2000);
