import os
from generate_dataset import clean_up_data, generate_dataset

CLIENT_NUMBER = 4
# generate_dataset(CLIENT_NUMBER, 0)

cmd_list = [f' python client.py {i} 0  > client-{i}-fault.log' for i in range(CLIENT_NUMBER)]
os.system('&'.join(cmd_list)) 