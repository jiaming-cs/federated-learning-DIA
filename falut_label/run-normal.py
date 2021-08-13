import os

CLIENT_NUMBER = 4
cmd_list = [f' python client.py {i} -1  > client-{i}-normal.log' for i in range(CLIENT_NUMBER)]
os.system('&'.join(cmd_list)) 