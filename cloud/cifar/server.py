from statistics import mode
import flwr as fl
import os
import pickle
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
from models import get_naive_cnn

# model = Sequential()
# model.add(InputLayer(input_shape=(128, )))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(10, activation='softmax'))

model = get_naive_cnn()
model.compile("adam", tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])


workspace_dir="./"
splited_data_folder="datasets"
with open(os.path.join(workspace_dir, splited_data_folder, f'data_test.pkl'), 'rb') as f:
    data = pickle.load(f)
    
    x_test, y_test = data['x_data'], data['y_data']
    # x_test = np.reshape(x_test, (-1, 128))
    # y_test = to_categorical(y_test, 10)
    
def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""


    x_val, y_val = x_test, y_test

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights):
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        print("Gobal Model Acc: ", accuracy)
        return loss, {"accuracy": accuracy}

    return evaluate


strategy = fl.server.strategy.FedAvg(fraction_fit=1, fraction_eval=1.0, min_fit_clients=2, min_available_clients=2, eval_fn=get_eval_fn(model))

fl.server.start_server(config={"num_rounds": 30}, strategy=strategy)
