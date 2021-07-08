import pickle

with open("./logs/iid-625/history-0.pkl", "rb") as f:
    data = pickle.load(f)
    print(data['test']['f1'][-1])