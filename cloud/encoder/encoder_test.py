from encoder import data_feature
import tensorflow as tf
from tensorflow.keras.utils import to_categorical



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

