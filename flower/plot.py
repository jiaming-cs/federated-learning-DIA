import matplotlib.pyplot as plt
import pickle

with open('./flower/history.plk', 'rb') as f:
    history = pickle.load(f)


plt.figure()

ep = len(history['train']['loss'])
plt.plot(range(ep), history['train']['loss'], label='LSTM training loss')
plt.plot(range(ep), history['val']['loss'], label='LSTM validation loss')


plt.legend()

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Variation')
plt.savefig('./img/'+'Loss_LSTM_detection.png')


plt.figure()

ep = len(history['train']['acc'])
plt.plot(range(ep), history['train']['acc'], label='LSTM training acc')
plt.plot(range(ep), history['val']['acc'], label='LSTM validation acc')


plt.legend()

plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.title('Acc Variation')
plt.savefig('./img/'+'Acc_LSTM_detection.png')
