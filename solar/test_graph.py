import os
import pickle
import matplotlib.pyplot as plt


plt.figure(1)

iid_folder = './logs/iid-625'
with open(os.path.join(iid_folder, f'history-0.pkl'), 'rb') as f:
    history = pickle.load(f)

ep = len(history['test']['acc'])
plt.subplot(211)
plt.plot(range(ep), history['test']['acc'], 'r', label='LSTM Test Acc')
plt.ylim(0.5, 1)
# plt.xlabel('Communication Rounds')
plt.ylabel('Acc')
plt.twinx()
plt.plot(range(ep), history['test']['loss'], 'b', label='LSTM Test Loss')
plt.ylim(0, 0.006)
plt.legend(loc=5)
# plt.xlabel('Communication Rounds')
plt.ylabel('Loss')
plt.title('Test Accuracy and Loss')


noniid_folder = './logs/non-iid-6-26'
with open(os.path.join(noniid_folder, f'history-0.pkl'), 'rb') as f:
    history = pickle.load(f)

ep = len(history['test']['acc'])
plt.subplot(212)
plt.plot(range(ep), history['test']['acc'], 'r', label='LSTM Test Acc')
plt.ylim(0.5, 1)
plt.legend(loc=5)
# plt.xlabel('Communication Rounds')
plt.ylabel('Acc')
plt.twinx()
plt.plot(range(ep), history['test']['loss'], 'b', label='LSTM Test Loss')
plt.ylim(0, 0.006)
# plt.legend()
plt.xlabel('Communication Rounds')
plt.ylabel('Loss')
# plt.title('Test Accuracy and Loss')

# import matplotlib.pyplot as plt
# import numpy as np
plt.show()



# def f(t):
#     return np.exp(-t) * np.cos(2 * np.pi * t)

# if __name__ == '__main__' :
#     t1 = np.arange(0, 5, 0.1)
#     t2 = np.arange(0, 5, 0.02)

#     plt.figure(1)
#     plt.subplot(121)
#     plt.plot(t1, f(t1), 'bo', t2, f(t2), 'r--')

#     plt.subplot(122)
#     plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')


#     plt.show()