import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, AveragePooling2D, Dense, Flatten, Input, Add, Lambda
from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import SGD, Adam
from sklearn.metrics import accuracy_score



def res_net():
    input_data = Input(shape=(xdim, num_features, 1))
    l = 0
    for i in range(15):
        if i == 0:
            x = Conv2D(45, kernel_size=(3,3), activation='relu', data_format='channels_last', 
                       padding='same', kernel_initializer='glorot_uniform')(input_data)
            x = BatchNormalization(axis=-1)(x)
            l += 1
            x = Conv2D(45, kernel_size=(3,3), activation='relu', data_format='channels_last', 
                       padding='same', kernel_initializer='glorot_uniform', dilation_rate=int(math.pow(2, np.floor(l/3))))(x)
            l += 1
            x = BatchNormalization(axis=-1)(x)
            
        else:
            y = Conv2D(45, kernel_size=(3,3), activation='relu', data_format='channels_last', 
                       padding='same', kernel_initializer='glorot_uniform', dilation_rate=int(math.pow(2, np.floor(l/3))))(x)
            y = BatchNormalization(axis=-1)(y)
            l += 1
            y = Conv2D(45, kernel_size=(3,3), activation='relu', data_format='channels_last', 
                       padding='same', kernel_initializer='glorot_uniform', dilation_rate=int(math.pow(2, np.floor(l/3))))(y)
            l += 1
            y = BatchNormalization(axis=-1)(y)
            
            y = Add()([y, x])
            x = Lambda(lambda x: x)(y)


    x = AveragePooling2D(pool_size=(2,2),data_format='channels_last')(x)
    x = Flatten()(x)
    x = Dense(units=len(keywords) + 1, activation='softmax')(x)
    
    model = Model(inputs=input_data, outputs=x) 
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    
    return model



def dnn_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(xdim, num_features, 1), data_format='channels_last')) 
    model.add(BatchNormalization(axis=-1))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu')) 
    model.add(BatchNormalization(axis=-1))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu')) 
    model.add(BatchNormalization(axis=-1))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    
    #model.add(Conv2D(256, kernel_size=(5,5), activation='relu')) 
    #model.add(BatchNormalization(axis=-1))
    #model.add(AveragePooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=len(keywords) + 1, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model