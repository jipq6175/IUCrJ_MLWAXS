
"""

The neural network architectures for different datasets

Specify the hyper-parameters and dataset and get training started

"""



import time
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization
from keras.optimizers import Adam
from pickle import dump

# Define the convolutional neural networks architecture
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


# Training 
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

