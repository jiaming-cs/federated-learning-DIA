import matplotlib.pyplot as plt

fed_02 = [0.894, 0.885, 0.879, 0.861, 0.841]
fed_03 = [0.897, 0.886, 0.870, 0.844, 0.803]
fed_04 = [0.896, 0.879, 0.867, 0.822, 0.730]
fed_05 = [0.899, 0.879, 0.855, 0.8220, 0.686]

attk = [0, 0.2, 0.4, 0.6, 0.8]

plt.plot(attk, fed_02, linestyle='-', color='b', label='FedAvg-0.2')
plt.plot(attk, fed_03, linestyle='-', color='g', label='FedAvg-0.3')
plt.plot(attk, fed_04, linestyle='-', color='r', label='FedAvg-0.4')
plt.plot(attk, fed_05, linestyle='-', color='c', label='FedAvg-0.5')

plt.xticks(attk)
plt.legend()
plt.ylim(.65, 0.9)
plt.xlabel('Fraction of malicious devices')
plt.ylabel('Test Accuracy')
plt.title('Global Model Test Accuray on FashionMNIST for FedAvg (20Client)')
plt.show()

plt.savefig('./img/result_fedavg.png')

kmean_02 = [0.878, 0.875, 0.875, 0.877, 0.872]
kmean_03 = [0.876, 0.878, 0.876, 0.877, 0.873]
kmean_04 = [0.876, 0.878, 0.874, 0.872, 0.869]
kmean_05 = [0.875, 0.874, 0.871, 0.870, 0.869]

plt.plot(attk, kmean_02, linestyle='-', color='b', label='Ours-0.2')
plt.plot(attk, kmean_03, linestyle='-', color='g', label='Ours-0.3')
plt.plot(attk, kmean_04, linestyle='-', color='r', label='Ours-0.4')
plt.plot(attk, kmean_05, linestyle='-', color='c', label='Ours-0.5')
plt.xticks(attk)
plt.legend()
plt.ylim(.8, 0.9)
plt.xlabel('Fraction of malicious devices')
plt.ylabel('Test Accuracy')
plt.title('Global Model Test Accuray on FashionMnist for Proposed Method (20Client)')
plt.show()