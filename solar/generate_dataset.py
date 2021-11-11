import numpy as np
import os
import pickle


def random_change(old):
    if old == 1:
        return 0
    else:
        return 1

def generate_dataset(x_train, y_train, x_test, y_test, client_num, falut_client_index, falut_ratio=0.2, workspace_dir="./", splited_data_folder="datasets"):
   
    clean_up_data()
    os.makedirs(os.path.join(workspace_dir, splited_data_folder))
    
        
    num_pre_client = x_train.shape[0] // client_num
    
    for i in range(client_num):
        with open(os.path.join(workspace_dir, splited_data_folder, f'data_{i}.pkl'), 'wb') as f:
            x_data = x_train[i*num_pre_client:i*num_pre_client+num_pre_client]
            y_data = y_train[i*num_pre_client:i*num_pre_client+num_pre_client] 
            pickle.dump(dict(x_data=x_data, y_data=y_data), f)
            
        if i<= falut_client_index:
            with open(os.path.join(workspace_dir, splited_data_folder, f'data_{i}_fault.pkl'), 'wb') as f:
                x_data = x_train[i*num_pre_client:i*num_pre_client+num_pre_client]
                y_data = y_train[i*num_pre_client:i*num_pre_client+num_pre_client] 
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
        
