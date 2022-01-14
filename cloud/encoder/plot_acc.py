import matplotlib.pyplot as plt




fed = [0.897, 0.872, 0.855, 0.795, 0.698]

kmean = [0.876, 0.872, 0.869, 0.868, 0.855]


attk = [0, 0.2, 0.4, 0.6, 0.8]


plt.plot(attk, fed, marker='*', color='r', label='FedAvg')
plt.plot(attk, kmean, marker='^', color='b', label='Ours')
plt.xticks(attk)
plt.legend()
plt.ylim(.6, 1)
plt.xlabel('Fraction of malicious devices')
plt.ylabel('Test Accuracy')
plt.title('Fasion MNIST')
plt.show()

plt.savefig('./img/Fasion-MNIST.png')


# fed_cifar = [0.663, 0.650, 0.546, 0.298, 0.275]

# kmean_cifar = [0.600, 0.591, 0.545, 0.533, 0.525]


# attk = [0, 0.2, 0.4, 0.6, 0.8]


# plt.plot(attk, fed_cifar, marker='*', color='r', label='FedAvg')
# plt.plot(attk, kmean_cifar, marker='^', color='b', label='Ours')
# plt.xticks(attk)
# plt.legend()
# plt.ylim(.0, 1)
# plt.xlabel('Fraction of malicious devices')
# plt.ylabel('Test Accuracy')
# plt.title('CIFAR-10')
# plt.show()

# plt.savefig('./img/CIFAR-10.png')

