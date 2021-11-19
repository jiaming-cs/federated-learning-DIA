import tensorflow as tf
from generate_dataset import generate_dataset
import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

CLIENT_NUMBER = 4
FAULT_INDEX = 1
FAULT_RATIO = 0.8

IS_KMEANS = True
if IS_KMEANS:
    EXP_NAME = f'{CLIENT_NUMBER}_{FAULT_INDEX+1}_attack_{FAULT_RATIO}_kmeans'
else:
    EXP_NAME = f'{CLIENT_NUMBER}_{FAULT_INDEX+1}_attack_{FAULT_RATIO}'

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# x_train, y_train = x_train[5000:], y_train[5000:]

# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
y_train = np.squeeze(y_train)
x_test = x_test.astype("float32") / 255.0
y_test = np.squeeze(y_test)

# x_train, x_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1)



# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)


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


# # x_train, x_test = data_feature(x_train), data_feature(x_test)



# datagen_test = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True)


# x_test = np.expand_dims(x_test, 3)

# datagen.fit(x_test)


# x_test_arg = []
# y_test_arg = []
# batches = 0
# for x_batch, y_batch in datagen_test.flow(x_test, y_test, batch_size=32):
#     x_test_arg.append(x_batch)
#     y_test_arg.append(y_batch)
#     batches += 1
#     if batches >= len(x_train) / 32:
#         # we need to break the loop by hand because
#         # the generator loops indefinitely
#         break

# x_test = np.concatenate(x_test_arg, axis=0)
# y_test = np.concatenate(y_test_arg, axis=0).flatten()
# print(y_test.shape)
# print(x_test.shape)



try:
    os.makedirs(f"./logs/{EXP_NAME}")
except:
    pass

def generate_cmd(fault_index=FAULT_INDEX, model_type='cnn', exp_type='local', exp_name = EXP_NAME, client_num = CLIENT_NUMBER):

    if IS_KMEANS:
        cmds = [f'python client.py -c {i} -f {fault_index} -m {model_type} -e {exp_type} -n {exp_name} -k > ./logs/{exp_name}/client_{i}_fault_{fault_index}.log' for i in range(client_num)]
    else:
        cmds = [f'python client.py -c {i} -f {fault_index} -m {model_type} -e {exp_type} -n {exp_name} > ./logs/{exp_name}/client_{i}_fault_{fault_index}.log' for i in range(client_num)]
    print(' & '.join(cmds))
    return ' & '.join(cmds)


# if len(x_train.shape) < 4:
#     x_train = np.expand_dims(x_train, -1)
#     x_test = np.expand_dims(x_test, -1)
    

generate_dataset(x_train, y_train, x_test, y_test, CLIENT_NUMBER, FAULT_INDEX, falut_ratio=FAULT_RATIO)


# os.system(f"nohup python server.py > ./logs/{EXP_NAME}/server.log 2>&1 &")

time.sleep(20)

os.system(generate_cmd())


# os.system("nohup python server.py > server.log 2>&1 &")

# "python client.py -c 0 -f -1 -m cnn -e local & python client.py -c 1 -f -1 -m cnn -e local"