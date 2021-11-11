from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D, Flatten
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np

import pickle
import os






def get_naive_cnn():
    _input = Input((100, 6, 1))
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(_input)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(256, activation='sigmoid')(x)
    x = Dense(128, name='encode', activation='sigmoid')(x)
    x = Dense(2, activation='softmax')(x)
    return Model(inputs=[_input], outputs=[x])



if __name__ == "__main__":
    with open(os.path.join('./', 'detection.pkl'), 'rb') as f:
        data = pickle.load(f)
        
    x_train = data['x_data']
    y_train = data['y_data']


    x_train, y_train = x_train[:10000], y_train[:10000]

    # model = get_naive_cnn_cifar()

    model = get_naive_cnn()

    y_train = to_categorical(y_train, 2)


    x_train = np.expand_dims(x_train, 3)

    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=10, batch_size=128)
    # print(model.evaluate(x_test, y_test))
    model.save('./weights/encoder_weights.h5')