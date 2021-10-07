import matplotlib.pyplot as plt
import pickle
import os

folder = './logs/test-iid'
client_num = 4

plt.figure()
for i in range(client_num):
    with open(os.path.join(folder, f'history-{i}.pkl'), 'rb') as f:
        history = pickle.load(f)
    
    plt.subplot(2, 2, i+1)
    ep = len(history['train']['loss'])
    plt.ylim(0, 0.006)
    plt.plot(range(ep), history['train']['loss'], label='Training Loss')
    plt.plot(range(len(history['test']['loss'][:-1])), history['test']['loss'][:-1], label='Test Loss')
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
    plt.plot(range(ep), history['train']['acc'], label='Training Acc')
    plt.plot(range(len(history['test']['acc'][:-1])), history['test']['acc'][:-1], label='Test Acc')
    plt.legend()
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy')
    plt.title(f'Client-{i} Accuracy')

plt.tight_layout() 
plt.savefig("./img/acc.png")
plt.figure()

