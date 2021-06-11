import os
import pickle
import numpy as np

path = ".\\PELS_dataset\\w100_final_dataset\\fault_detection\\deep_learning\\"
normal_path = path + 'Normal/'
abnormal_path = path + 'Abnormal/'

normal_files = [f for f in os.listdir(normal_path) if f.endswith('.csv')]
abnormal_files = [f for f in os.listdir(abnormal_path) if f.endswith('.csv')]

np_normal = []
np_abnormal = []

for f in normal_files:
    temp_file = np.loadtxt(normal_path + f, delimiter=',', dtype=np.float32)
    np_normal.append(temp_file)

for f in abnormal_files:
    temp_file = np.loadtxt(abnormal_path + f, delimiter=',', dtype=np.float32)
    np_abnormal.append(temp_file)
    
# x: input data, y: target
y_data = np.concatenate([np.zeros(len(np_normal),dtype=np.long), \
                                np.zeros(len(np_abnormal),dtype=np.long)+1])
x_data = np.concatenate([np_normal, np_abnormal])

with open("detection.pkl", 'wb') as f:
    print("save")
    pickle.dump(dict(x_data=x_data, y_data=y_data), f)
    