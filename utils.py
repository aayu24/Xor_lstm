import argparse

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences


def generate_samples(length=50):
    '''
    Generate random binary strings of variable lenght
    Args: length-length of string
    Returns: numpy array of binary strings and array of parity bit labels
    '''

    if length == 50:
        data = np.random.randint(2, size=(50000, length)).astype('float32')
        labels = [0 if sum(i) % 2 == 0 else 1 for i in data]
    else:
        data = []
        labels = []
        for i in range(50000):
            length = np.random.randint(1, 51)
            data.append(np.random.randint(2, size=(length)).astype('float32'))
            labels.append(0 if sum(data[i]) % 2 == 0 else 1)
        data = np.asarray(data)
        # Pad binary strings with 0's to make sequence length sma for all
        data = pad_sequences(data, maxlen=50, dtype='float32', padding='pre')

    labels = np.asarray(labels, dtype='float32')
    train_size = data.shape[0]
    print(data.shape)
    print(train_size)
    print(length)
    size = int(train_size * 0.75)

    # Splitting data in train and test sets
    X_train = data[:size]
    X_test = data[size:]
    y_train = labels[:size]
    y_test = labels[size:]

    # Expanding dimensions of data set to feed into lstm layer
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    return X_train, y_train, X_test, y_test

def build_model():
    '''Build LSTM model using Keras
       Args: none
       Returns: Compiled LSTM model
    '''
    model = Sequential()
    model.add(LSTM(32, input_shape=(50, 1)))
    model.add(Dense(1, activation='sigmoid'))
    # Display summary of model
    model.summary()
    model.compile('adam', loss='binary_crossentropy', metrics=['acc'])
    return model

def model_plot(history):
    '''
    Plot models acuracy and loss
    Args: history-Keras dictionary containing training/validation loss/accuracy
    Returns: plots model's training/validation loss with accuracy history
    '''
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training_loss')
    plt.plot(epochs, val_loss, 'b', label='Validation_loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()

    plt.figure()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.plot(epochs, acc, 'bo', label='Training_accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation_accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()

    plt.show()
    return
