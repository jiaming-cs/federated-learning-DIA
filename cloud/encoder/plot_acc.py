import matplotlib.pyplot as plt

# attk = [0, 0.2, 0.4, 0.6]

# fedavg_1_attk = [0.817, 0.807, 0.804, 0.798]

# fedavg_2_attk = [0.817, 0.795, 0.785, 0.760]

# fedavg_3_attk = [0.817, 0.786, 0.733, 0.663]



# kmean_2_attk = [0.786, 0.785, 0.786, 0.784]

# kmean_3_attk = [0.786, 0.785, 0.782, 0.781]

# attk_portion = [0, 0.25, 0.5, 0.75]
# fed_60 = [0.817, 0.798, 0.760, 0.663]
# kmean_60 = [0.786, 0.783, 0.784, 0.781]




# plt.plot(attk_portion, fed_60, marker='*', color='r', label='FedAvg')
# plt.plot(attk_portion, kmean_60, marker='^', color='b', label='Kmeans')
# plt.xticks(attk_portion)
# plt.legend()

# plt.xlabel('Fraction of malicious devices')
# plt.ylabel('Test Accuracy')
# plt.title('FEMINIST')
# plt.show()

# plt.savefig('./img/FEMINIST_60.png')


fed = [0.897, 0.872, 0.855, 0.795, 0.698]

kmean = [0.876, 0.872, 0.869, 0.868, 0.855]


attk = [0, 0.2, 0.4, 0.6, 0.8]


plt.plot(attk, fed, marker='*', color='r', label='FedAvg')
plt.plot(attk, kmean, marker='^', color='b', label='Kmeans')
plt.xticks(attk)
plt.legend()
plt.ylim(.6, 1)
plt.xlabel('Fraction of malicious devices')
plt.ylabel('Test Accuracy')
plt.title('FashionMnist-20Client-0.8Corrupt')
plt.show()

plt.savefig('./img/FashionMnist-20Client-0.8Corrupt.png')

