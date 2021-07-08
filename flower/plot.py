import matplotlib.pyplot as plt
import pickle
import os

folder = './logs/iid-625'
client_num = 4

plt.figure()
for i in range(client_num):
    with open(os.path.join(folder, f'history-{i}.pkl'), 'rb') as f:
        history = pickle.load(f)
    
    plt.subplot(2, 2, i+1)
    ep = len(history['train']['loss'])
    plt.ylim(0, 0.006)
    plt.plot(range(ep), history['train']['loss'], label='LSTM training loss')
    plt.plot(range(ep), history['val']['loss'], label='LSTM validation loss')
    plt.legend()
    plt.xlabel('Communication Rounds')
    plt.ylabel('Loss')
    plt.title(f'Client-{i} Loss')
plt.tight_layout() 
plt.savefig("./img/lose.png")
plt.figure()

for i in range(client_num):
    with open(os.path.join(folder, f'history-{i}.pkl'), 'rb') as f:
        history = pickle.load(f)

    plt.subplot(2, 2, i+1)
    ep = len(history['train']['acc'])
    plt.ylim(0.5, 1)
    plt.plot(range(ep), history['train']['acc'], label='LSTM training acc')
    plt.plot(range(ep), history['val']['acc'], label='LSTM validation acc')
    plt.legend()
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy')
    plt.title(f'Client-{i} Accuracy')

plt.tight_layout() 
plt.savefig("./img/acc.png")
plt.figure()

with open(os.path.join(folder, f'history-0.pkl'), 'rb') as f:
    history = pickle.load(f)

ep = len(history['test']['acc'])
plt.plot(range(ep), history['test']['acc'], 'r', label='LSTM Test Acc')
plt.ylim(0.5, 1)
plt.legend()
plt.xlabel('Communication Rounds')
plt.ylabel('Acc')
plt.twinx()
plt.plot(range(ep), history['test']['loss'], 'b', label='LSTM Test Loss')
plt.ylim(0, 0.006)
plt.legend()
plt.xlabel('Communication Rounds')
plt.ylabel('Loss')
plt.title('Test Accuracy and Loss')
plt.savefig("./img/test.png")