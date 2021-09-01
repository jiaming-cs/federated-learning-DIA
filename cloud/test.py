import pickle

with open(r'C:\Code\Summer2021\federated-learning-DIA\cloud\datasets/data_3.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)