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


fed = [0.90, 0.88, 0.84, 0.77, 0.63]

kmean = [0.90, 0.87, 0.87, 0.87, 0.86]


attk = [0, 0.2, 0.4, 0.6, 0.8]


plt.plot(attk, fed, marker='*', color='r', label='FedAvg')
plt.plot(attk, kmean, marker='^', color='b', label='Kmeans')
plt.xticks(attk)
plt.legend()
plt.ylim(.5, 1)
plt.xlabel('Fraction of malicious devices')
plt.ylabel('Test Accuracy')
plt.title('FashionMnist-10Client-0.8Corrupt')
plt.show()

plt.savefig('./img/FashionMnist-10Client-0.8Corrupt.png')


# fed = [0.87, 0.84, 0.74, 0.58]
# kmean_8 = [0.80, 0.79, 0.77, 0.78]
# kmean_2 = [0.79, 0.77, 0.78, 0.79]



# attk = [0, 0.4, 0.6, 0.8]


# plt.plot(attk, fed, marker='*', color='r', label='FedAvg')
# plt.plot(attk, kmean_8, marker='^', color='b', label='Kmeans-0.8error')
# plt.plot(attk, kmean_2, marker='.', color='g', label='Kmeans-0.2error')
# plt.xticks(attk)
# plt.legend()
# plt.ylim(.5, 1)
# plt.xlabel('Fraction of malicious devices')
# plt.ylabel('Test Accuracy')
# plt.title('FashionMnist-10Client')
# plt.show()

# plt.savefig('./img/FashionMnist-10Client-0.8Corrupt.png')


# plt.plot(attk, fedavg_2_attk, marker='*', color='r', label='FedAvg')
# plt.plot(attk, kmean_2_attk, marker='^', color='b', label='Kmeans')

# plt.legend()

# plt.xlabel('Portion of data being attacked for each client')
# plt.ylabel('Golbal model testing accuracy')
# plt.title('Accuracy comparison (2 corrupt clients)')
# plt.show()

# plt.savefig('./img/two_corrupt.png')



# plt.plot(attk, fedavg_3_attk, marker='*', color='r', label='FedAvg')
# plt.plot(attk, kmean_3_attk, marker='^', color='b', label='Kmeans')

# plt.legend()

# plt.xlabel('Portion of data being attacked for each client')
# plt.ylabel('Golbal model testing accuracy')
# plt.title('Accuracy comparison (3 corrupt clients)')
# plt.show()

# plt.savefig('./img/three_corrupt.png')