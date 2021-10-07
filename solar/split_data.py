import pickle
import os
from sklearn.model_selection import train_test_split

import numpy as np

def get_distribution(client_num, factor=5):
    ret = []
    while not ret or min(ret)<0.05 or max(ret) < factor * min(ret):
        ret = [np.random.rand() for _ in range(client_num)]
    return [n/sum(ret) for n in ret]


def split_data(input_data, client_num, falut_client_index, falut_ratio=0.8, workspace_dir='./', splited_data_folder='splited_data'):
    os.makedirs(os.path.join(workspace_dir, splited_data_folder))
    with open(os.path.join(workspace_dir, input_data), 'rb') as f:
        data = pickle.load(f)
    x_data = data['x_data']
    y_data = data['y_data']
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)
    
    num_pre_client = x_train.shape[0] // client_num
    
    fault_client_index_list = list(range(falut_client_index+1))
    
    for i in range(client_num):
        with open(os.path.join(workspace_dir, splited_data_folder, f'data_{i}.pkl'), 'wb') as f:
            x_data = x_train[i*num_pre_client:i*num_pre_client+num_pre_client].copy()
            y_data = y_train[i*num_pre_client:i*num_pre_client+num_pre_client].copy()
            pickle.dump(dict(x_data=x_data, y_data=y_data), f)
            
        if i in fault_client_index_list:
            with open(os.path.join(workspace_dir, splited_data_folder, f'data_{i}_fault.pkl'), 'wb') as f:
                x_data = x_train[i*num_pre_client:i*num_pre_client+num_pre_client].copy()
                y_data = y_train[i*num_pre_client:i*num_pre_client+num_pre_client].copy() 
                fault_num = int(y_data.shape[0]*falut_ratio)
                fault_index = np.random.choice(range(y_data.shape[0]), fault_num)
                for j in fault_index:
                    y_data[j] = 0 if y_data[j] == 1 else 0
                pickle.dump(dict(x_data=x_data, y_data=y_data), f)
        print(f"create data {i}")
                
                
    with open(os.path.join(workspace_dir, splited_data_folder, f'test_data.pkl'), 'wb') as f:  
        pickle.dump(dict(x_data=x_test, y_data=y_test), f)
    
    
        
def clean_up_data(workspace_dir='./', splited_data_folder='splited_data'):
    if os.path.exists(os.path.join(os.path.join(workspace_dir, splited_data_folder))):
        for f in os.listdir(os.path.join(os.path.join(workspace_dir, splited_data_folder))):
            os.remove(os.path.join(workspace_dir, splited_data_folder, f))
        os.removedirs(os.path.join(workspace_dir, splited_data_folder))

# clean_up_data()

# split_data('detection.pkl', 5)