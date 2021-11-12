import numpy as np
import os
import pickle

LABEL_NUM = 10
def random_change(old, label_num=LABEL_NUM):
    new = int(np.random.rand() * label_num)
    while new == old:
        new = int(np.random.rand() * label_num)
    return new

def generate_dataset(x_train, y_train, x_test, y_test, client_num, falut_client_index, falut_ratio=0.8, workspace_dir="./", splited_data_folder="datasets"):
   
    clean_up_data()
    os.makedirs(os.path.join(workspace_dir, splited_data_folder))

    num_pre_client = x_train.shape[0] // client_num
    
    fault_client_index_list = list(range(falut_client_index+1))
    
    for i in range(client_num):
        with open(os.path.join(workspace_dir, splited_data_folder, f'data_{i}.pkl'), 'wb') as f:
            x_data = x_train[i*num_pre_client:i*num_pre_client+num_pre_client].copy()
            y_data = y_train[i*num_pre_client:i*num_pre_client+num_pre_client].copy()
            # x_data = np.reshape(x_data, (-1, 128))
            pickle.dump(dict(x_data=x_data, y_data=y_data), f)
            
        if i in fault_client_index_list:
            with open(os.path.join(workspace_dir, splited_data_folder, f'data_{i}_fault.pkl'), 'wb') as f:
                x_data = x_train[i*num_pre_client:i*num_pre_client+num_pre_client].copy()
                y_data = y_train[i*num_pre_client:i*num_pre_client+num_pre_client].copy() 
                fault_num = int(y_data.shape[0]*falut_ratio)
                
                fault_index = np.random.choice(y_data.shape[0], fault_num, replace=False)
                print(fault_index)
                for j in fault_index:
                    y_data[j] = np.random.randint(0, 10)
                # x_data = np.reshape(x_data, (-1, 128))
                pickle.dump(dict(x_data=x_data, y_data=y_data), f)
        print(f"create data {i}")
                
                
    with open(os.path.join(workspace_dir, splited_data_folder, f'data_test.pkl'), 'wb') as f:  
        pickle.dump(dict(x_data=x_test, y_data=y_test), f)
            
            
def clean_up_data(workspace_dir='./', splited_data_folder='datasets'):
    if os.path.exists(os.path.join(os.path.join(workspace_dir, splited_data_folder))):
        for f in os.listdir(os.path.join(os.path.join(workspace_dir, splited_data_folder))):
            os.remove(os.path.join(workspace_dir, splited_data_folder, f))
        os.removedirs(os.path.join(workspace_dir, splited_data_folder))
        
