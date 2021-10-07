from tensorflow.keras.models import Model, load_model

model = load_model("./weights/encoder_weights.h5")

data_feature = Model(inputs=model.input, outputs=model.layers[-2].output)
