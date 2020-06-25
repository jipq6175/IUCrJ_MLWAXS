
# Prepare data for training the convolutional neural networks
# dependencies
import os
import numpy as np
import h5py as h5


# The function to load hdf5 files into tensors for training data
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

