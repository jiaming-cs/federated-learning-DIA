from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np

# # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
fault_ratio = 0.5
fault_num = int(y_train.shape[0]*fault_ratio)
                
fault_index = np.random.choice(range(y_train.shape[0]), fault_num)

y_train = list(y_train)
for j in fault_index:
    y_train[j] = np.random.randint(0, 10)


x_train, y_train = x_train[:5000], y_train[:5000]


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


def get_naive_cnn_cifar():
    _input = Input((32, 32, 3))
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





# model = get_naive_cnn_cifar()

model = get_naive_cnn()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = np.expand_dims(x_train, 3)

model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=50, batch_size=128)
print(model.evaulate(x_test, y_test))
model.save('./weights/encoder_weights.h5')