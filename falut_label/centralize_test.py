import tensorflow as tf
from tensorflow.keras.utils import to_categorical
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


y_train = y_train.flatten()
print(y_train.shape)
y_test = y_test.flatten()

y_train = to_categorical(y_train, 10)
print(y_train.shape)
y_test = to_categorical(y_test, 10)


model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)


model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=20, batch_size=128)

loss, accuracy = model.evaluate(x_test, y_test)
        

print(f"Test Loss: {loss}, Test Acc: {accuracy}")