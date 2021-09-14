from tensorflow.keras import models
from tensorflow.python import module
from encoder_model import get_naive_cnn
from tensorflow.keras.models import Model
model = get_naive_cnn()

model.load_weights("./weights/encoder_weights.h5")

data_feature = Model(inputs=model.input, outputs=model.get_layer("endcode").output)
