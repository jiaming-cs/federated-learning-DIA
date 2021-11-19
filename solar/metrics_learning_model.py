import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import defaultdict
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras import layers
from collections import Counter

from tensorflow.keras.datasets import fashion_mnist
import os
import pickle
from sklearn.model_selection import train_test_split

# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# x_train = x_train.astype("float32")[:5000] / 255.0
# y_train = np.squeeze(y_train)[:5000]
# x_test = x_test.astype("float32")[:5000] / 255.0
# y_test = np.squeeze(y_test)[:5000]
# ct = Counter(y_train)
# print(ct.most_common())


with open(os.path.join('.', 'detection.pkl'), 'rb') as f:
    data = pickle.load(f)
    
x_data = data['x_data']
y_data = data['y_data']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)
        
x_train = x_train / np.max(x_train)
x_test = x_test/ np.max(x_test)


x_train, x_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1)

# height_width = 28


# def show_collage(examples):
#     box_size = height_width + 2
#     num_rows, num_cols = examples.shape[:2]

#     collage = Image.new(
#         mode="RGB",
#         size=(num_cols * box_size, num_rows * box_size),
#         color=(250, 250, 250),
#     )
#     for row_idx in range(num_rows):
#         for col_idx in range(num_cols):
#             array = (np.array(examples[row_idx, col_idx]) * 255).astype(np.uint8)
#             collage.paste(
#                 Image.fromarray(array), (col_idx * box_size, row_idx * box_size)
#             )

#     # Double size for visualisation.
#     collage = collage.resize((2 * num_cols * box_size, 2 * num_rows * box_size))
#     return collage


# # Show a collage of 5x5 random images.
# sample_idxs = np.random.randint(0, 50000, size=(5, 5))
# examples = x_train[sample_idxs]
# show_collage(examples)

class_idx_to_train_idxs = defaultdict(list)
for y_train_idx, y in enumerate(y_train):
    class_idx_to_train_idxs[y].append(y_train_idx)

class_idx_to_test_idxs = defaultdict(list)
for y_test_idx, y in enumerate(y_test):
    class_idx_to_test_idxs[y].append(y_test_idx)
    
num_classes = 2


class AnchorPositivePairs(keras.utils.Sequence):
    def __init__(self, num_batchs):
        self.num_batchs = num_batchs

    def __len__(self):
        return self.num_batchs

    def __getitem__(self, _idx):
        x = np.empty((2, num_classes, 100, 6, 1), dtype=np.float32)
        for class_idx in range(num_classes):
            examples_for_class = class_idx_to_train_idxs[class_idx]
            anchor_idx = random.choice(examples_for_class)
            positive_idx = random.choice(examples_for_class)
            while positive_idx == anchor_idx:
                positive_idx = random.choice(examples_for_class)
            x[0, class_idx] = x_train[anchor_idx]
            x[1, class_idx] = x_train[positive_idx]
        return x

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
            sparse_labels = tf.range(num_classes)
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

model = EmbeddingModel(inputs, embeddings)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

history = model.fit(AnchorPositivePairs(num_batchs=1000), epochs=20)

model.save_weights('./weights/encoder_weights.h5')
# near_neighbours_per_example = 10

# embeddings = model.predict(x_test)

# print(embeddings)
