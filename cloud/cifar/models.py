from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D, Flatten
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf
import numpy as np

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# x_train = x_train.astype("float32") / 255.0
# y_train = np.squeeze(y_train)
# x_test = x_test.astype("float32") / 255.0
# y_test = np.squeeze(y_test)


mobile_net = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)

def get_naive_cnn():
    _input = Input((32, 32, 3))
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(_input)
    x = MaxPool2D((2, 2))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPool2D((2, 2))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10)(x)
    return Model(inputs=[_input], outputs=[x])


def get_ann():
    model = Sequential()
    model.add(Input(shape=(128, )))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    print('output shape:', model.output_shape)
    return model

# y_train = tf.keras.utils.to_categorical(y_train, 10)
# y_test = tf.keras.utils.to_categorical(y_test, 10)

# model = get_naive_cnn()

# model.compile("adam", tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

# model.fit(x_train, y_train, epochs=10, batch_size=256)

# model.evaluate(x_test, y_test)