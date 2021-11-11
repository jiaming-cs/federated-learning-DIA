import flwr as fl
import numpy as np
from tensorflow.keras.utils import to_categorical
from argparse import ArgumentParser
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf
import os
from models import get_naive_cnn, mobile_net
from kmeans import fit_kmeans


paser = ArgumentParser()

paser.add_argument('--client_index', '-c', type=int)
paser.add_argument('--fault_index', '-f', type=int)
paser.add_argument('--model_type', '-m', type=str)
paser.add_argument('--exp_type', '-e', type=str)
paser.add_argument('--exp_name', '-n', type=str)
paser.add_argument('-k', action='store_true')

args = paser.parse_args()


client_index = args.client_index
fault_index = args.fault_index
model_type = args.model_type
exp_type = args.exp_type
exp_name = args.exp_name

is_kmeans = args.k


print('exp_name', exp_name)

history = {'train': {'loss': [], 'acc': []}, 'val': {'loss': [], 'acc': []},
           'test': {'loss': [], 'acc': [], 'f1': [], 'recall': [], 'precision': []}}

def load_data(client_index, fault_client_index, workspace_dir="./", splited_data_folder="datasets"):
    if client_index == fault_client_index:
        with open(os.path.join(workspace_dir, splited_data_folder, f'data_{client_index}_fault.pkl'), 'rb') as f:
            data = pickle.load(f)
            x_train, y_train = data['x_data'], data['y_data']
            print("data num:", len(y_train))
        with open(os.path.join(workspace_dir, splited_data_folder, f'data_{client_index}.pkl'), 'rb') as f:
            data = pickle.load(f)
            _, y_train_gt = data['x_data'], data['y_data']
    else:
        with open(os.path.join(workspace_dir, splited_data_folder, f'data_{client_index}.pkl'), 'rb') as f:
            data = pickle.load(f)
            x_train, y_train = data['x_data'], data['y_data']
    with open(os.path.join(workspace_dir, splited_data_folder, f'data_{client_index}.pkl'), 'rb') as f:
        data = pickle.load(f)
        x_test, y_test = data['x_data'], data['y_data']

    if is_kmeans:
        x_train_kmeans = data_feature(x_train)
        kmeans_selected = fit_kmeans(x_train_kmeans, y_train)
        x_train, y_train = x_train[kmeans_selected], y_train[kmeans_selected]
        print("after kmeans:", len(y_train))
        if y_train_gt is not None:
            y_selected = y_train_gt[kmeans_selected]
            ct = Counter(y_selected)
            print(ct.most_common())
        else:
            
            ct = Counter(y_train)
            print(ct.most_common())
    print(f'x_train:{x_train.shape} y_train:{len(y_train)}')
    print(f'x_test:{x_test.shape} y_test:{len(y_test)}')

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

print(f"client_index:{client_index}")
print(f"fault_index:{fault_index}")

x_train, y_train, x_test, y_test = load_data(client_index=client_index, fault_client_index=fault_index)

print("x_train", x_train.shape)
print("y_train", y_train.shape)

print("x_test", x_test.shape)
print("y_test", y_test.shape)

if model_type == "mobile_net":
    model = mobile_net
else:
    model = get_naive_cnn()

model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])


class CifarClient(fl.client.NumPyClient):

    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        h = model.fit(x_train, y_train, epochs=1, batch_size=128)
        print('history:', h)
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

if exp_type == 'cloud':
    fl.client.start_numpy_client("10.142.0.3:8080", client=CifarClient())
else:
    fl.client.start_numpy_client("[::]:8080", client=CifarClient())
    # fl.client.start_numpy_client("localhost:8080", client=CifarClient())

print(history)

with open(f'./logs/{exp_name}/history-{client_index}-fault-{fault_index}.pkl', 'wb') as f:
    pickle.dump(history, f)
