import os
from split_data import split_data, clean_up_data

clean_up_data()


CLIENT_NUMBER = 2
split_data('detection.pkl', CLIENT_NUMBER)

cmd_list = [f' python client.py {i} ' for i in range(CLIENT_NUMBER)]
os.system('&'.join(cmd_list)) 