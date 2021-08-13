import flwr as fl
import numpy as np
from model import LSTM
from loader import WaveformDetectionDLPickle
from torch.utils.data import DataLoader
import torch
from torch import nn
from collections import OrderedDict
from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt
import time
import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score

DATASET_DIR = './splited_data'
INPUT_SIZE = 6 # num of feature for deep learning
EPOCH = 1
BATCH_SIZE = 128
TIME_STEP = 100  # length of LSTM time sequence, or window size
VAL_SPLIT = 0.1
LR = 0.001   # learning rate
isGPU = torch.cuda.is_available()
loss_func = nn.CrossEntropyLoss()

history = {'train':{'loss':[], 'acc':[]}, 'val':{'loss':[], 'acc':[]}, 'test':{'loss':[], 'acc':[], 'f1':[], 'recall':[], 'precision':[]}}

def load_data(client_num):
    train_dataset = WaveformDetectionDLPickle(DATASET_DIR, client_num)
    test_dataset = WaveformDetectionDLPickle(DATASET_DIR, -1)
    
    return train_dataset, test_dataset

def test(lstm, test_dataset):
    lstm.eval()
    lstm_eval_loss = 0
    lstm_eval_acc = 0
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)
    lstm_final_prediction = np.array([])
    lstm_final_test = np.array([])
    for step, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.view(-1, TIME_STEP, INPUT_SIZE)
        
        if isGPU:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

        output_lstm = lstm(batch_x)
        batch_y = batch_y.type(torch.LongTensor)
        loss_lstm = loss_func(output_lstm, batch_y)
        
        lstm_eval_loss += loss_lstm.item()
        
        lstm_pred = torch.max(output_lstm, 1)[1]
        
        lstm_eval_correct = (lstm_pred == batch_y).sum()
        
        
        if isGPU:
            lstm_pred = torch.max(output_lstm, 1)[1].cuda()
        else:
            lstm_pred = torch.max(output_lstm, 1)[1]
        
        lstm_eval_acc += lstm_eval_correct.item()
        lstm_final_prediction = np.concatenate((lstm_final_prediction, lstm_pred.cpu().numpy()), axis=0)
        lstm_final_test = np.concatenate((lstm_final_test, batch_y), axis=0)

    f1 = f1_score(lstm_final_test, lstm_final_prediction, average='binary').item()
    recall = recall_score(lstm_final_test, lstm_final_prediction, average='binary').item()
    precision = precision_score(lstm_final_test, lstm_final_prediction, average='binary').item()
    lstm_eval_acc /= float(len(test_loader.dataset))
    lstm_eval_loss /= float(len(test_loader.dataset))
    history['test']['acc'].append(lstm_eval_acc)
    history['test']['loss'].append(lstm_eval_loss)
    history['test']['f1'].append(f1)
    history['test']['recall'].append(recall)
    history['test']['precision'].append(precision)
    print(f'Testing Loss: {lstm_eval_loss}, Accuracy: {lstm_eval_acc}, f1: {f1}, recall: {recall}, precision: {precision}')
    return lstm_eval_loss, lstm_eval_acc

def train(lstm, train_dataset, test_dataset, epochs):
    """Train the network on the training set."""
    
    if isGPU:
        lstm = nn.DataParallel(lstm, device_ids=[0,1,2,3])
        lstm.cuda()

    lstm_optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    training_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(epochs):
        lstm_train_loss = 0
        
        lstm_train_acc = 0
        
        lstm.train()
        
        for step, (batch_x, batch_y) in enumerate(training_loader):
            batch_x = batch_x.view(-1, TIME_STEP, INPUT_SIZE)
            
            if isGPU:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            
            output_lstm = lstm(batch_x)
            
            batch_y = batch_y.type(torch.LongTensor)
          
            loss_lstm = loss_func(output_lstm, batch_y)
            
            lstm_train_loss += loss_lstm.item()

            if isGPU:
                lstm_pred = torch.max(output_lstm, 1)[1].cuda()
            else:
                lstm_pred = torch.max(output_lstm, 1)[1]
            
            lstm_train_correct = (lstm_pred == batch_y).sum()
            
            lstm_train_acc += lstm_train_correct.item()
            
            lstm_optimizer.zero_grad()
            
            loss_lstm.backward()
            
            lstm_optimizer.step()
        
        lstm_train_acc /= float(len(training_loader.dataset))
        lstm_train_loss /= float(len(training_loader.dataset))
        history['train']['acc'].append(lstm_train_acc)
        history['train']['loss'].append(lstm_train_loss)

        

        print(f'Epoch:{epoch}')
        print(f'Train Loss: {lstm_train_loss}, Accuracy: {lstm_train_acc}')
        


paser = ArgumentParser()

paser.add_argument('client_num', type=int)

args = paser.parse_args()

net = LSTM()
train_dataset, test_dataset = load_data(args.client_num)

class CifarClient(fl.client.NumPyClient):

        
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, train_dataset, test_dataset, epochs=EPOCH)
        return self.get_parameters(), len(train_dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, test_dataset)
        return float(loss), len(test_dataset), {"accuracy":float(accuracy)}
    

fl.client.start_numpy_client("localhost:8080", client=CifarClient())
print(history)

with open(f'./history-{args.client_num}.pkl', 'wb') as f:
    pickle.dump(history, f)    

plt.figure()

ep = len(history['train']['loss'])
print(history)
plt.plot(range(len(history['train']['loss'])), history['train']['loss'], label='Training loss')
plt.plot(range(len(history['test']['loss'])), history['test']['loss'], label='Testing loss')


plt.legend()

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Variation')
plt.savefig('./'+'Loss_LSTM_detection.png')


plt.figure()

ep = len(history['train']['acc'])
plt.plot(range(len(history['train']['acc'])), history['train']['acc'], label='Training acc')
plt.plot(range(len(history['test']['acc'])), history['test']['acc'], label='Testing acc')


plt.legend()

plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.title('Acc Variation')
plt.savefig('./'+'Acc_LSTM_detection.png')



