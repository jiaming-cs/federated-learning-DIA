from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D, Flatten
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# fault_ratio = 0.5
# fault_num = int(y_train.shape[0]*fault_ratio)
                
# fault_index = np.random.choice(range(y_train.shape[0]), fault_num)

# y_train = list(y_train)
# for j in fault_index:
#     y_train[j] = np.random.randint(0, 10)


# x_train, y_train = x_train[:5000], y_train[:5000]

latent_dim = 64 

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(784, activation='sigmoid'),
      layers.Reshape((28, 28))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded




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

# x_train, y_train = x_train[:5000], y_train[:5000]

# model = get_naive_cnn()


# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     validation_split=0.2)


# x_train = np.expand_dims(x_train, 3)

# datagen.fit(x_train)


# x_train_arg = []
# y_train_arg = []
# batches = 0
# for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
#     x_train_arg.append(x_batch)
#     y_train_arg.append(y_batch)
#     batches += 1
#     if batches >= 5 * (len(x_train) / 32):
#         # we need to break the loop by hand because
#         # the generator loops indefinitely
#         break

# x_train = np.concatenate(x_train_arg, axis=0)
# y_train = np.concatenate(y_train_arg, axis=0).flatten()
# print(y_train.shape)
# print(x_train.shape)
# y_train = to_categorical(y_train)
# # y_test = to_categorical(y_test)
if __name__ == '__main__':
    

    # x_train, y_train = x_train[:5000], y_train[:5000]
    
    # model = get_naive_cnn()
    
    model = Autoencoder(latent_dim)
    # datagen = ImageDataGenerator(
    #     featurewise_center=True,
    #     featurewise_std_normalization=True,
    #     rotation_range=20,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     horizontal_flip=True,
    #     validation_split=0.2)


    # x_train = np.expand_dims(x_train, 3)
    # y_train = to_categorical(y_train)

# datagen.fit(x_train)
    model.compile("adam", "mean_squared_error", metrics=["accuracy"])
    model.fit(x_train, x_train, epochs=30, batch_size=128)
    # print(model.evaluate(x_test, y_test))
    model.save_weights('./weights/encoder_weights.h5')