import os
os.system("python server.py & sleep 3 & python run-fault.py")
os.system("python server.py & sleep 3 & python run-normal.py")
