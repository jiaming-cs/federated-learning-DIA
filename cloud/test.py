import pickle

with open(r'C:\Code\Summer2021\federated-learning-DIA\cloud\logs/with_attack/history-0-fault-0.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)