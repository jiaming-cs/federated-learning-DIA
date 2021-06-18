import pickle

with open("./logs/history-1623741657.024014.pkl", "rb") as f:
    data = pickle.load(f)
    print(data)