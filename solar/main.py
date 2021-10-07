import os

from split_data import split_data, clean_up_data

clean_up_data()


CLIENT_NUMBER = 4
FAULT_INDEX = 1
split_data('detection.pkl', CLIENT_NUMBER, FAULT_INDEX)

cmd_list = [f' python client.py {i} {FAULT_INDEX} > client-{i}.log' for i in range(CLIENT_NUMBER)]
os.system('&'.join(cmd_list)) 