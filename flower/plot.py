import matplotlib.pyplot as plt
import pickle
import os

folder = './non-iid-6-21'
client_num = 4

plt.figure()

for i in range(client_num):
    with open(os.path.join(folder, f'history-{i}.pkl'), 'rb') as f:
        history = pickle.load(f)
    
    plt.subplot(2, 2, i+1)
    ep = len(history['train']['loss'])
    plt.plot(range(ep), history['train']['loss'], label='LSTM training loss')
    plt.plot(range(ep), history['val']['loss'], label='LSTM validation loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Variation Client-{i}')
    
plt.savefig("./img/lose.png")

plt.figure()

for i in range(client_num):
    with open(os.path.join(folder, f'history-{i}.pkl'), 'rb') as f:
        history = pickle.load(f)

    plt.subplot(2, 2, i+1)
    ep = len(history['train']['acc'])
    plt.plot(range(ep), history['train']['acc'], label='LSTM training acc')
    plt.plot(range(ep), history['val']['acc'], label='LSTM validation acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.title(f'Acc Variation Client-{i}')

plt.savefig("./img/acc.png")

plt.figure()
with open(os.path.join(folder, f'history-0.pkl'), 'rb') as f:
    history = pickle.load(f)

ep = len(history['test']['acc'])
plt.plot(range(ep), history['test']['acc'], label='LSTM Test Acc')
plt.plot(range(ep), history['test']['loss'], label='LSTM Test Loss')
plt.legend()
plt.xlabel('Communication Rounds')
plt.ylabel('Acc/loss')
plt.title('Acc/Loss Variation')
plt.savefig("./img/test.png")