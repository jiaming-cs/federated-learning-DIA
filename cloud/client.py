import flwr as fl
import numpy as np
from tensorflow.keras.utils import to_categorical
from argparse import ArgumentParser
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf
import os

history = {'train':{'loss':[], 'acc':[]}, 'val':{'loss':[], 'acc':[]}, 'test':{'loss':[], 'acc':[], 'f1':[], 'recall':[], 'precision':[]}}

def load_data(client_index, fault_client_index, workspace_dir="./", splited_data_folder="datasets"):    
    if client_index == fault_client_index:
        with open(os.path.join(workspace_dir, splited_data_folder, f'data_{client_index}_fault.pkl'), 'rb') as f:
            data = pickle.load(f)
            x_train, y_train = data['x_data'], data['y_data']
    else:
        with open(os.path.join(workspace_dir, splited_data_folder, f'data_{client_index}.pkl'), 'rb') as f:
            data = pickle.load(f)
            x_train, y_train = data['x_data'], data['y_data']
    with open(os.path.join(workspace_dir, splited_data_folder, f'data_{client_index}.pkl'), 'rb') as f:
        data = pickle.load(f)
        x_test, y_test = data['x_data'], data['y_data']    
        
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)   
    return x_train, y_train, x_test, y_test
    


paser = ArgumentParser()

paser.add_argument('client_index', type=int)
paser.add_argument('fault_index', type=int)

args = paser.parse_args()
client_index, fault_index = args.client_index, args.fault_index

print(f"client_index:{client_index}")
print(f"fault_index:{fault_index}")
        
x_train, y_train, x_test, y_test = load_data(client_index=client_index, fault_client_index=fault_index)

print("x_train", x_train.shape)
print("y_train", y_train.shape)

print("x_test", x_test.shape)
print("y_test", y_test.shape)

model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])


class CifarClient(fl.client.NumPyClient):
 
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        h = model.fit(x_train, y_train, epochs=1, batch_size=128)
        print(h.history.keys())
        loss, acc = h.history['loss'][0], h.history['accuracy'][0]
        print(f"Train Loss: {loss}, Train Acc: {acc}")
        history['train']['loss'].append(loss)
        history['train']['acc'].append(acc)
        
        loss, accuracy = model.evaluate(x_test, y_test)
        history['test']['loss'].append(loss)
        history['test']['acc'].append(accuracy)
        
        print(f"Test Loss: {loss}, Test Acc: {accuracy}")
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}
    

fl.client.start_numpy_client("10.142.0.3:8080", client=CifarClient())
print(history)

with open(f'./history-{client_index}-fault-{fault_index}.pkl', 'wb') as f:
    pickle.dump(history, f)    


