# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 01:03:52 2018

@author: ashkp
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


# coding: utf-8

# In[2]:


import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.layers import Bidirectional
from keras.preprocessing.sequence import pad_sequences


# In[18]:


def generate_samples(length=50):
    '''Generate random binary strings of variable lenght
           Args: length-length of string
           Returns: numpy array of binary strings and array of parity bit labels
    '''
    if length==50:
        data=np.random.randint(2,size=(50000,length)).astype('float32')
        labels=[0 if sum(i)%2==0 else 1 for i in data]
    else:
        data=[]
        labels=[]
        for i in range(50000):
            length=np.random.randint(1,51)
            data.append(np.random.randint(2,size=(length)).astype('float32'))
            labels.append(0 if sum(data[i])%2==0 else 1)
        data=np.asarray(data)
        data=pad_sequences(data,maxlen=50,dtype='float32',padding='post')
        #pad binary strings with 0's to make sequence length sma for all
        #data = pad_sequences(data,maxlen=50,dtype='float32',padding='pre')
        
    labels=np.asarray(labels,dtype='float32')
    train_size=data.shape[0]
    print(data.shape)
    print(train_size)
    print(length)
    size=int(train_size*0.75)

    #splitting data in train and test sets
    X_train=data[:size]
    X_test=data[size:]
    y_train=labels[:size]
    y_test=labels[size:]

    #expanding dimensions of data set to feed into lstm layer
    X_train=np.expand_dims(X_train,axis=2)
    X_test=np.expand_dims(X_test,axis=2)

    return X_train,y_train,X_test,y_test


# In[19]:


def build_model():
    '''Build LSTM model using Keras
       Args: none
       Returns: Compiled LSTM model
    '''
    model=Sequential()
    #model.add(LSTM(2,input_shape=(50,1)))
    model.add(Bidirectional(LSTM(2,return_sequences=False),input_shape=(50,1)))
    model.add(Dense(1,activation='sigmoid'))
    #display summary of model
    model.summary()
    model.compile('adam',loss='binary_crossentropy',metrics=['acc'])
    return model


# In[20]:


def model_plot(history):
    '''Plot models acuracy and loss
           Args: history-Keras dictionary containing training/validation loss/accuracy
           Returns: plots model's training/validation loss with accuracy history 
    '''
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    
    epochs = range(1,len(loss)+1)
    plt.figure()
    plt.plot(epochs,loss,'bo',label='Training_loss')
    plt.plot(epochs,val_loss,'b',label='Validation_loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    
     
    plt.figure()
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    plt.plot(epochs,acc,'bo',label='Training_accuracy')
    plt.plot(epochs,val_acc,'b',label='Validation_accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    
    plt.show()
    return


# In[24]:


def main(length=50):
    '''Build and train LSTM network to solve XOR problem'''

    X_train,y_train,X_test,y_test = generate_samples(length=length)
    model = build_model()
    history=model.fit(X_train,y_train,epochs=30,batch_size=32,validation_split=0.10,shuffle=False)

    #evaluate model on test set
    preds=model.predict(X_test)
    preds=np.round(preds[:,0]).astype('float32')
    acc= (np.sum(preds==y_test)/len(y_test))*100
    print('Accuracy: {:.2f}%'.format(acc))
    
    #plotting loss and accuracy
    model_plot(history)
    return


# In[23]:


#Command Line Arguments
if __name__ == '__main__':
    '''Execute main program'''
    
    #set seed
    np.random.seed(32)
    #Grab user arguments
    parser=argparse.ArgumentParser()
    parser.add_argument('-l','--length',help='define binary string length (40 or -1)')
    
    args=parser.parse_args()
    if args.length=='50':
        print("Generating binary strings of length 50")
        main(length=50)
    elif args.length=='-1':
        print("Generating binary strings of length b/w 1 and 50")
        main(length=-1)
    else:
        print('Invalid entry')
