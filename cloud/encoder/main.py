import tensorflow as tf
from generate_dataset import generate_dataset
import numpy as np
import time
from encoder import data_feature

import os

CLIENT_NUMBER = 4
FAULT_INDEX = 3
EXP_NAME = '4_attack_80_error_centralized_fashion'
IS_KMEANS = False

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train, y_train = x_train[5000:], y_train[5000:]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train, x_test = data_feature(x_train), data_feature(x_test)



print(x_train.shape)
print(x_test.shape)

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


if len(x_train.shape) < 4:
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    

generate_dataset(x_train, y_train, x_test, y_test, CLIENT_NUMBER, FAULT_INDEX)


os.system(f"nohup python server.py > ./logs/{EXP_NAME}/server.log 2>&1 &")

time.sleep(20)

os.system(generate_cmd())


# os.system("nohup python server.py > server.log 2>&1 &")

# "python client.py -c 0 -f -1 -m cnn -e local & python client.py -c 1 -f -1 -m cnn -e local"