from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers import LSTM, Input, TimeDistributed,Convolution2D,Activation
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import RMSprop, SGD
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import optimizers
from keras.preprocessing import sequence
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import load_model

def alexnet(input_shape,num_classes): 
    model = Sequential() 
    # Layer 1 
    model.add(Convolution2D(96, 11, 11, input_shape = input_shape, border_mode='same')) 
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    # Layer 2 
    model.add(Convolution2D(128, 5, 5, border_mode='same')) 
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2)) 
    # Layer 3 
    model.add(ZeroPadding2D((1,1))) 
    model.add(Convolution2D(384, 3, 3, border_mode='same')) 
    model.add(Activation('relu')) 
    # Layer 4 
    model.add(ZeroPadding2D((1,1))) 
    model.add(Convolution2D(192, 3, 3, border_mode='same')) 
    model.add(Activation('relu'))
    # Layer 5 
    model.add(ZeroPadding2D((1,1))) 
    model.add(Convolution2D(128, 3, 3, border_mode='same')) 
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    # Layer 6 
    model.add(GlobalAveragePooling2D()) 
    model.add(Dense(4096, init='glorot_normal')) 
    model.add(Activation('relu')) 
    model.add(Dropout(0.5)) 
    # Layer 7 
    model.add(Dense(4096, init='glorot_normal')) 
    model.add(Activation('relu')) 
    model.add(Dropout(0.5)) 
    # Layer 8 
    model.add(Dense(num_classes, init='glorot_normal')) 
    model.add(Activation('tanh')) 
    return model