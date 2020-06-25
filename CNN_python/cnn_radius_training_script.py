# script for training the helical_radius

import sys
sys.path.append('G:/My Drive/14. CNNWAXS/data');
from prepare_data import *
from training import *


# Set up data directory
datadir = 'G:/My Drive/14. CNNWAXS/data/helical_radius';
os.chdir(datadir);

# Load the data
(x_train, y_train) = load_helix_training_data(datadir);


#
# Data transformation
(mi, ma, interval, y_train_new) = label_transform(y_train);
(x_train_sample, y_train_sample) = uniform_sampling(x_train, y_train_new);
x_train_sample = np.reshape(x_train_sample, (*x_train_sample.shape, 1));
y_train_sample = np_utils.to_categorical(y_train_sample, 25);



#
# set up the model and train 
model = convolutional_neural_nets(indim=191, outdim=25, n_filters=64, conv_kernel=10, conv_stride=1, pool_kernel=8, pool_stride=2);

#
model, hist = train_model(model, 'helical_radius', x_train_sample, y_train_sample, learning_rate=2.5e-4, batchsize=256, n_epochs=20);

