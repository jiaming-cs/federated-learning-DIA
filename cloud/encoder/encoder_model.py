from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np

(_, _), (x_train, y_train) = tf.keras.datasets.mnist.load_data()



def get_naive_cnn():
    _input = Input((28, 28, 1))
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(_input)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, name='encode', activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    return Model(inputs=[_input], outputs=[x])


model = get_naive_cnn()

y_train = to_categorical(y_train)

x_train = np.expand_dims(x_train, 3)

model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=50, batch_size=128)
model.save('./weights/encoder_weights.h5')