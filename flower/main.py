import os
from split_data import split_data, clean_up_data

clean_up_data()


CLIENT_NUMBER = 4
split_data('detection.pkl', CLIENT_NUMBER, iid=False)

cmd_list = [f' python client.py {i} > client-{i}.log' for i in range(CLIENT_NUMBER)]
os.system('&'.join(cmd_list)) 