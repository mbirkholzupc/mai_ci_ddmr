#Source material
#https://faroit.com/keras-docs/1.2.2/layers/convolutional/
#https://arxiv.org/pdf/1602.07261v1.pdf
#https://towardsdatascience.com/review-inception-v4-evolved-from-googlenet-merged-with-resnet-idea-image-classification-5e8c339d18bc


import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.engine import merge, Input, Model
import keras.backend as K

#stem schema
def inception_v4_stem(x):

    x = Convolution2D(32, 3, 3, strides=(2, 2), activation='relu')(x)
    x = Convolution2D(32, 3, 3, stride=(1, 1), activation='relu')(x)
    x = Convolution2D(64, 3, 3, stride=(1, 1), activation='relu')(x)
    

    a = MaxPooling2D((3, 3), strides=(2, 2))(x)

    b = Convolution2D(96, 3, 3, strides=(2, 2), activation='relu')(x)
    x = merge([a, b], mode='concat', concat_axis=-1)
    
    a = Convolution2D(64, 1, 1, stride=(1, 1), activation='relu')(x)
    a = Convolution2D(96, 3, 3, stride=(1, 1), activation='relu')(a)
    b = Convolution2D(64, 1, 1, stride=(1, 1), activation='relu')(x)
    b = Convolution2D(64, 7, 1, stride=(1, 1), activation='relu')(b)
    b = Convolution2D(64, 1, 7, stride=(1, 1), activation='relu')(b)
    b = Convolution2D(96, 3, 3, stride=(1, 1), activation='relu')(b)
    x = merge([a, b], mode='concat', concat_axis=-1)
    
    a = Convolution2D(192, 3, 3, stride=(1, 1), activation='relu')(x)
    b = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = merge([a, b], mode='concat', concat_axis=-1)
    
    return x

#schema for 35x35
def inception_v4_A(x):
    a = AveragePooling2D((3, 3), strides=(1, 1))(x)
    a = Convolution2D(96, 1, 1, stride=(1, 1), activation='relu')(a)
    
    b = Convolution2D(96, 1, 1, stride=(1, 1), activation='relu')(x)
    
    c = Convolution2D(64, 1, 1, stride=(1, 1), activation='relu')(x)
    c = Convolution2D(96, 3, 3, stride=(1, 1), activation='relu')(c)
    
    d = Convolution2D(64, 1, 1, stride=(1, 1), activation='relu')(x)
    d = Convolution2D(96, 3, 3, stride=(1, 1), activation='relu')(d)
    d = Convolution2D(96, 3, 3, stride=(1, 1), activation='relu')(d)
    
    x = merge([a, b, c, d], mode='concat', concat_axis=-1)
    
    return x
#35x35 to 17x17 reduction schema
#Network Inception-v4 k=192, l=224, m=256, n=384
#c path c1=k, c2=l, c3=m
def inception_v4_reduction_A(x):
    a = MaxPooling2D((3, 3), strides=(2, 2))(x)
    b = Convolution2D(384, 3, 3, stride=(2, 2), activation='relu')(x)
    c = Convolution2D(192, 1, 1, stride=(1, 1), activation='relu')(x)
    c = Convolution2D(224, 3, 3, stride=(1, 1), activation='relu')(c)
    c = Convolution2D(256, 3, 3, stride=(2, 2), activation='relu')(c)
    
    x = merge([a, b, c], mode='concat', concat_axis=-1)
    
    return x
    
#17x17 schema
def inception_v4_B(x):
    a = AveragePooling2D((3, 3), strides=(1, 1))(x)
    a = Convolution2D(128, 1, 1, stride=(1, 1), activation='relu')(a)
    
    b = Convolution2D(384, 1, 1, stride=(1, 1), activation='relu')(x)
    
    c = Convolution2D(192, 1, 1, stride=(1, 1), activation='relu')(x)
    c = Convolution2D(224, 1, 7, stride=(1, 1), activation='relu')(c)
    c = Convolution2D(256, 1, 7, stride=(1, 1), activation='relu')(c)
    
    d = Convolution2D(192, 1, 1, stride=(1, 1), activation='relu')(x)
    d = Convolution2D(192, 1, 7, stride=(1, 1), activation='relu')(d)
    d = Convolution2D(224, 7, 1, stride=(1, 1), activation='relu')(d)
    d = Convolution2D(224, 1, 7, stride=(1, 1), activation='relu')(d)
    d = Convolution2D(256, 7, 1, stride=(1, 1), activation='relu')(d)
    
    x = merge([a, b, c, d], mode='concat', concat_axis=-1)
    
    return x

#17x17 to 8x8 reduction schema
def inception_v4_reduction_B(x):
    a = MaxPooling2D((3, 3), strides=(2, 2))(x)
    b = Convolution2D(192, 1, 1, stride=(1, 1), activation='relu')(x)
    b = Convolution2D(192, 3, 3, stride=(2, 2), activation='relu')(b)
    c = Convolution2D(256, 1, 1, stride=(1, 1), activation='relu')(x)
    c = Convolution2D(256, 1, 7, stride=(1, 1), activation='relu')(c)
    c = Convolution2D(320, 7, 1, stride=(1, 1), activation='relu')(c)
    c = Convolution2D(320, 3, 3, stride=(2, 2), activation='relu')(c)
    
    x = merge([a, b, c], mode='concat', concat_axis=-1)
    
    return x

#8x8 schema
def inception_v4_C(x):
    a = AveragePooling2D((3, 3), strides=(1, 1))(x)
    a = Convolution2D(256, 1, 1, stride=(1, 1), activation='relu')(a)
    
    b = Convolution2D(256, 1, 1, stride=(1, 1), activation='relu')(x)
    
    c = Convolution2D(384, 1, 1, stride=(1, 1), activation='relu')(x)
    c1 = Convolution2D(256, 1, 3, stride=(1, 1), activation='relu')(c)
    c2 = Convolution2D(256, 3, 1, stride=(1, 1), activation='relu')(c)
    
    d = Convolution2D(384, 1, 1, stride=(1, 1), activation='relu')(x)
    d = Convolution2D(448, 1, 3, stride=(1, 1), activation='relu')(d)
    d = Convolution2D(512, 3, 1, stride=(1, 1), activation='relu')(d)
    d1 = Convolution2D(256, 3, 1, stride=(1, 1), activation='relu')(d)
    d2 = Convolution2D(256, 1, 3, stride=(1, 1), activation='relu')(d)
    
    x = merge([a, b, c1, c2, d1, d2], mode='concat', concat_axis=-1)
    
    return x

#model creation-----------------------------------------------------------------
def main_cnn(img_rows, img_cols, img_chls, num_class):    

    num_A_blocks = 1
    num_B_blocks = 1
    num_C_blocks = 1

    inputs = Input(shape=(img_rows, img_cols, img_chls))

    x = inception_v4_stem(inputs)
    for i in range(num_A_blocks):
        x = inception_v4_A(x)
    x = inception_v4_reduction_A(x)
    for i in range(num_B_blocks):
        x = inception_v4_B(x)
    x = inception_v4_reduction_B(x)
    for i in range(num_C_blocks):
        x = inception_v4_C(x)

    #average ploting and dropout layers
    x = AveragePooling2D(pool_size=(4, 4))(x)
    x = Dropout(0.8)(x)
    x = Flatten()(x)

    predictions = Dense(num_class, activation='softmax')(x)

    model = Model(input=inputs, output=predictions)

    #model summary----------------------------------------------------------------
    model.summary()

    #model compile---------------------------------------------------------------
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

    #model fit------------------------------------------------------------------
if __name__ == "__main__":
    history = model.fit_generator(data, steps_per_epoch=4,validation_data=train_data, valifatio_steps=4, epochs=5)