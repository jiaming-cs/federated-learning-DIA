import subprocess
import tensorflow as tf
from generate_dataset import generate_dataset
import numpy as np
import time

import os

CLIENT_NUMBER = 4
FAULT_INDEX = 0
EXP_NAME = 'with_attack'
IS_KMEANS = True
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

def generate_cmd(fault_index=FAULT_INDEX, model_type='cnn', exp_type='local', exp_name = EXP_NAME, client_num = CLIENT_NUMBER):
    os.makedirs(f"./logs/{exp_name}")
    if IS_KMEANS:
        cmds = [f'python client.py -c {i} -f {fault_index} -m {model_type} -e {exp_type} -k > ./logs/{exp_name}/client_{i}_fault_{fault_index}.log' for i in range(client_num)]
    else:
        cmds = [f'python client.py -c {i} -f {fault_index} -m {model_type} -e {exp_type} > ./logs/{exp_name}/client_{i}_fault_{fault_index}.log' for i in range(client_num)]
    return ' & '.join(cmds)


if len(x_train.shape) < 4:
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    

generate_dataset(x_train, y_train, x_test, y_test, CLIENT_NUMBER, FAULT_INDEX)


os.system("nohup python server.py > server.log 2>&1 &")

time.sleep(5)

os.system(generate_cmd())


# os.system("nohup python server.py > server.log 2>&1 &")

# "python client.py -c 0 -f -1 -m cnn -e local & python client.py -c 1 -f -1 -m cnn -e local"