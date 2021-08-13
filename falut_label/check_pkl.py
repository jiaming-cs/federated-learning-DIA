import pickle

with open("./falut_label/logs/normal_test/history-3-fault--1.pkl", "rb") as f:
    data = pickle.load(f)
    print(data['test']['f1'][-1])