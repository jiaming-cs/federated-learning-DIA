
from tensorflow.keras.models import Model, load_model


import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import defaultdict
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras import layers

# model = Autoencoder(64)
# model.build((128, 28, 28))
# model.summary()
# # model.load_weights("C:/Code/Summer2021/federated-learning-DIA/cloud/encoder/weights/encoder_weights.h5")
# model.load_weights("./weights/encoder_weights.h5")
# data_feature = model.encoder

# import tensorflow as tf
# import numpy as np

# # # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.

# out = data_feature(x_test)
# out = out.numpy()
# print(out)

class EmbeddingModel(keras.Model):
    def train_step(self, data):
        # Note: Workaround for open issue, to be removed.
        if isinstance(data, tuple):
            data = data[0]
        anchors, positives = data[0], data[1]

        with tf.GradientTape() as tape:
            # Run both anchors and positives through model.
            anchor_embeddings = self(anchors, training=True)
            positive_embeddings = self(positives, training=True)

            # Calculate cosine similarity between anchors and positives. As they have
            # been normalised this is just the pair wise dot products.
            similarities = tf.einsum(
                "ae,pe->ap", anchor_embeddings, positive_embeddings
            )

            # Since we intend to use these as logits we scale them by a temperature.
            # This value would normally be chosen as a hyper parameter.
            temperature = 0.2
            similarities /= temperature

            # We use these similarities as logits for a softmax. The labels for
            # this call are just the sequence [0, 1, 2, ..., num_classes] since we
            # want the main diagonal values, which correspond to the anchor/positive
            # pairs, to be high. This loss will move embeddings for the
            # anchor/positive pairs together and move all other pairs apart.
            sparse_labels = tf.range(2)
            loss = self.compiled_loss(sparse_labels, similarities)

        # Calculate gradients and apply via optimizer.
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return metrics (specifically the one for the loss value).
        self.compiled_metrics.update_state(sparse_labels, similarities)
        return {m.name: m.result() for m in self.metrics}
    
inputs = layers.Input(shape=(100, 6, 1))
x = layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding='same')(inputs)
x = layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding='same')(x)
# x = layers.Conv2D(filters=128, kernel_size=3, strides=2, activation="relu", padding='same')(x)
x = layers.GlobalAveragePooling2D()(x)
embeddings = layers.Dense(units=8, activation=None)(x)
embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

data_feature = EmbeddingModel(inputs, embeddings)

data_feature.load_weights("./weights/encoder_weights.h5")