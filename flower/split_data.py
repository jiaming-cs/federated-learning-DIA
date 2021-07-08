import pickle
import os
from sklearn.model_selection import train_test_split
import numpy as np

def get_distribution(client_num, factor=5):
    ret = []
    while not ret or min(ret)<0.05 or max(ret) < factor * min(ret):
        ret = [np.random.rand() for _ in range(client_num)]
    return [n/sum(ret) for n in ret]


def split_data(input_data, client_num, amount_non_iid = False, iid=True, abnormal_fraction = 0.75, workspace_dir='./', splited_data_folder='splited_data'):
    os.makedirs(os.path.join(workspace_dir, splited_data_folder))
    with open(os.path.join(workspace_dir, input_data), 'rb') as f:
        data = pickle.load(f)
    x_data = data['x_data']
    y_data = data['y_data']
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)
    with open(os.path.join(workspace_dir, splited_data_folder, f'test_data.pkl'), 'wb') as f:
        pickle.dump(dict(x_data=x_test, y_data=y_test), f)
        print(f"Test Set: P:{len(x_test[y_test==1])}, N:{len(x_test[y_test==0])}")
    
    if amount_non_iid:
        dist = get_distribution(client_num)
        train_length = x_train.shape[0]-1
        data_nums = [int(train_length * p) for p in dist]
        print("Data Numbers:", data_nums)
        for i in range(client_num):
            with open(os.path.join(workspace_dir, splited_data_folder, f'data_{i}.pkl'), 'wb') as f:
                x_data = x_train[sum(data_nums[:i]): sum(data_nums[:i])+data_nums[i]]
                y_data = y_train[sum(data_nums[:i]): sum(data_nums[:i])+data_nums[i]]
                pickle.dump(dict(x_data=x_data, y_data=y_data), f)                
                print(f"Train Set {i}: P:{len(x_data[y_data==1])}, N:{len(x_data[y_data==0])}")
    else:
        if not iid:      
            x_data_normal, x_data_abnormal = x_train[y_train==0], x_train[y_train==1]
            num_normal_client_0, num_abnormal_client_0 = int(len(x_data_normal) * (1-abnormal_fraction)), int(len(x_data_abnormal) * abnormal_fraction)
            num_normal_other, num_abnormal_other = (len(x_data_normal) - num_normal_client_0) // (client_num - 1), (len(x_data_abnormal) - num_abnormal_client_0) // (client_num - 1)
            
            for i in range(client_num):
                with open(os.path.join(workspace_dir, splited_data_folder, f'data_{i}.pkl'), 'wb') as f:
                    if i == 0:
                        x_data = np.concatenate([x_data_abnormal[:num_abnormal_client_0], x_data_normal[:num_normal_client_0]])
                        y_data = np.concatenate([np.ones((num_abnormal_client_0,)), np.zeros((num_normal_client_0,))])
                        pickle.dump(dict(x_data=x_data, y_data=y_data), f)
                        print(f"Train Set {i}: P:{len(x_data[y_data==1])}, N:{len(x_data[y_data==0])}")
                        x_data_abnormal, x_data_normal = x_data_abnormal[num_abnormal_client_0:], x_data_normal[num_normal_client_0:]
                    
                    else:
                        
                        x_data = np.concatenate([x_data_abnormal[(i-1)*num_abnormal_other: i*num_abnormal_other], x_data_normal[(i-1)*num_normal_other: i*num_normal_other]])
                        y_data = np.concatenate([np.ones((num_abnormal_other,)), np.zeros((num_normal_other,))])
                        pickle.dump(dict(x_data=x_data, y_data=y_data), f)
                        print(f"Train Set {i}: P:{len(x_data[y_data==1])}, N:{len(x_data[y_data==0])}")
        else:
            

            data_num = len(x_train) // client_num
            
            for i in range(client_num):
                with open(os.path.join(workspace_dir, splited_data_folder, f'data_{i}.pkl'), 'wb') as f:
                    x_data, y_data=x_train[i*data_num: i*data_num+data_num], y_train[i*data_num:i*data_num+data_num]
                    pickle.dump(dict(x_data=x_data, y_data=y_data), f)
                    
                    print(f"Train Set {i}: P:{len(x_data[y_data==1])}, N:{len(x_data[y_data==0])}")
                print(f"split data {i}")
        
def clean_up_data(workspace_dir='./', splited_data_folder='splited_data'):
    if os.path.exists(os.path.join(os.path.join(workspace_dir, splited_data_folder))):
        for f in os.listdir(os.path.join(os.path.join(workspace_dir, splited_data_folder))):
            os.remove(os.path.join(workspace_dir, splited_data_folder, f))
        os.removedirs(os.path.join(workspace_dir, splited_data_folder))

# clean_up_data()

# split_data('detection.pkl', 10)