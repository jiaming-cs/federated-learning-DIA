import flwr as fl
import tensorflow as tf
import threading
import os
import pickle
from loader import WaveformDetectionDLPickle
CLIENT_NUM = 4

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

train_num, test_num = x_train.shape[0] // CLIENT_NUM, x_test.shape[0] // CLIENT_NUM

model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])






class CifarClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        super().__init__()
        self.client_id = client_id
        
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        local_x_train = x_train[self.client_id * train_num: (self.client_id + 1) * train_num]
        local_y_train = y_train[self.client_id * train_num: (self.client_id + 1) * train_num]
        model.fit(local_x_train, local_y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        local_x_test = x_test[self.client_id * test_num: (self.client_id + 1) * test_num]
        local_y_test = y_test[self.client_id * test_num: (self.client_id + 1) * test_num]
        loss, accuracy = model.evaluate(local_x_test, local_y_test)
        return loss, len(x_test), {"accuracy": accuracy}
    
def start_client(client_id):
    fl.client.start_numpy_client("[::]:8080", client=CifarClient(client_id))

for i in range(CLIENT_NUM):
    p = Process(target = start_client, args = (i, ))
    p.start()
    


