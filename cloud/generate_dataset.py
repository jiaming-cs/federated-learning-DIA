import tensorflow as tf
import numpy as np
import os
import pickle

LABEL_NUM = 10
def random_change(old, label_num=LABEL_NUM):
    new = int(np.random.rand() * label_num)
    while new == old:
        new = int(np.random.rand() * label_num)
    return new

def generate_dataset(client_num, falut_client_index, falut_ratio=0.1, workspace_dir="./", splited_data_folder="datasets"):
   
    clean_up_data()
    os.makedirs(os.path.join(workspace_dir, splited_data_folder))
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train_grouped = []
    for i in range(LABEL_NUM):
        indices = [j for j, n in enumerate(y_train) if n==i]
        x_train_grouped.append(x_train[indices])
        
    num_pre_client = x_train.shape[0] // client_num
    num_pre_label =  num_pre_client // LABEL_NUM
    
    for i in range(client_num):
        with open(os.path.join(workspace_dir, splited_data_folder, f'data_{i}.pkl'), 'wb') as f:
            x_data = np.concatenate([data[i*num_pre_label: (i+1)*num_pre_label] for data in x_train_grouped])
            y_data = np.array([[j]*num_pre_label for j in range(LABEL_NUM)]).flatten()    
            pickle.dump(dict(x_data=x_data, y_data=y_data), f)
        if falut_client_index == i:
            with open(os.path.join(workspace_dir, splited_data_folder, f'data_{i}_fault.pkl'), 'wb') as f:
                x_data = np.concatenate([data[i*num_pre_label: (i+1)*num_pre_label] for data in x_train_grouped])
                y_data = np.array([[j]*num_pre_label for j in range(LABEL_NUM)]).flatten()   
                fault_num = int(y_data.shape[0]*falut_ratio)
                fault_index = np.random.choice(range(y_data.shape[0]), fault_num)
                for j in fault_index:
                    y_data[j] = random_change(y_data[j])
                pickle.dump(dict(x_data=x_data, y_data=y_data), f)
        print(f"create data {i}")
                
                
    with open(os.path.join(workspace_dir, splited_data_folder, f'data_test.pkl'), 'wb') as f:  
        pickle.dump(dict(x_data=x_test, y_data=y_test), f)
            
            
def clean_up_data(workspace_dir='./', splited_data_folder='datasets'):
    if os.path.exists(os.path.join(os.path.join(workspace_dir, splited_data_folder))):
        for f in os.listdir(os.path.join(os.path.join(workspace_dir, splited_data_folder))):
            os.remove(os.path.join(workspace_dir, splited_data_folder, f))
        os.removedirs(os.path.join(workspace_dir, splited_data_folder))
        
