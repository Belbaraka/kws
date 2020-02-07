import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, AveragePooling2D, Dense, Flatten, Input, Add, Lambda, MaxPooling2D, Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam
from sklearn.metrics import accuracy_score
from losses import *
import math
import numpy as np


def res_net(nb_keywords, xdim=98, num_features=40):
    
    input_data = Input(shape=(xdim, num_features, 1))
    l = 0
    #for i in range(6) conv filter 45:
    for i in range(3):
        if i == 0:
            x = Conv2D(20, kernel_size=(3,3), activation='relu', data_format='channels_last', 
                       padding='same', kernel_initializer='glorot_uniform')(input_data)
            x = BatchNormalization(axis=-1)(x)
            l += 1
            x = Conv2D(20, kernel_size=(3,3), activation='relu', data_format='channels_last', 
                       padding='same', kernel_initializer='glorot_uniform', dilation_rate=int(math.pow(2, np.floor(l/3))))(x)
            l += 1
            x = BatchNormalization(axis=-1)(x)
            
        else:
            y = Conv2D(20, kernel_size=(3,3), activation='relu', data_format='channels_last', 
                       padding='same', kernel_initializer='glorot_uniform', dilation_rate=int(math.pow(2, np.floor(l/3))))(x)
            y = BatchNormalization(axis=-1)(y)
            l += 1
            y = Conv2D(20, kernel_size=(3,3), activation='relu', data_format='channels_last', 
                       padding='same', kernel_initializer='glorot_uniform', dilation_rate=int(math.pow(2, np.floor(l/3))))(y)
            l += 1
            y = BatchNormalization(axis=-1)(y)
            
            y = Add()([y, x])
            x = Lambda(lambda x: x)(y)


    x = AveragePooling2D(pool_size=(2,2),data_format='channels_last')(x)
    x = Flatten()(x)
    x = Dense(units=nb_keywords + 1, activation='softmax')(x)
    
    model = Model(inputs=input_data, outputs=x) 
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    
    return model

def dnn_model(nb_keywords, xdim=98, num_features=40):
    model = Sequential()
    
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(xdim, num_features, 1), data_format='channels_last')) 
    model.add(BatchNormalization(axis=-1))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu')) 
    model.add(BatchNormalization(axis=-1))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, kernel_size=(3,3), activation='relu')) 
    model.add(BatchNormalization(axis=-1))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(units=128, activation='linear')) #32
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=128, activation='linear'))#32
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=256, activation='relu')) #128
    #model.add(Dropout(rate=0.2))
    model.add(Dense(units=nb_keywords + 1, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model


def cnn_parada(nb_keywords, xdim=98, num_features=40):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(20,8), activation='relu', input_shape=(xdim, num_features, 1), data_format='channels_last', strides=(1, 1))) 
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(1, 3)))
    
    model.add(Conv2D(64, kernel_size=(10,4), activation='relu')) 
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(1, 3)))
    
    model.add(Flatten())
    model.add(Dense(units=32, activation='linear'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=nb_keywords + 1, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    return model

def baseline_dnn(nb_keywords, xdim=98, num_features=40):
    model = Sequential()
    #model.add(Input(shape=(xdim, num_features, 1)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=nb_keywords + 1, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    return model