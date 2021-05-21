# -*- coding: utf-8 -*-
__author__ = "Alejandro Jerónimo Fuentes"
__date__ = "16/09/2020"


# %% Import neccesary packages

import json

import config.breast_histopathology_cancer_config as config
import matplotlib.pylab as plt
import numpy as np
import utilities.preprocessing as preprocessors
from keras.applications import InceptionV3
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from utilities.fclayer import FCInception
from utilities.kerasgenerator import KerasGenerator

# %% Load values RGB

values_rgb = json.loads(open(config.MEAN_PATH).read())


# %% Define preprocessors and generators

batch_size = 32
number_epochs = 20

aug = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=30,
    zoom_range=[0.5, 1.0],
)

rp = preprocessors.ResizePreprocessor(299, 299)
mrgb = preprocessors.MeanRGBPreprocessor(values_rgb)

trainGen = KerasGenerator(
    config.TRAIN_HDF5,
    batch_size,
    preprocessors=[rp, mrgb],
    data_augmentation=aug,
)
valGen = KerasGenerator(config.VAL_HDF5, batch_size, preprocessors=[rp, mrgb])

# %% Define base model

baseModel = InceptionV3(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(
        shape=(299, 299, 3),
    ),
)

head = FCInception.build(baseModel, 256)

model = Model(inputs=baseModel.input, outputs=head)

# %% Freeze every layer in original Inception architecture

for layer in baseModel.layers:
    layer.trainable = False

# %% Compile the new model

opt = Adam(lr=1e-4, decay=1e-4 / number_epochs)
model.compile(
    loss="categorical_crossentropy",
    optimizer=opt,
    metrics=["accuracy"],
)

# %% Training model

print("[INFO] Training Model")
H1 = model.fit(
    trainGen.generate_image_batch(),
    steps_per_epoch=trainGen.num_images // batch_size,
    validation_data=valGen.generate_image_batch(),
    validation_steps=valGen.num_images // batch_size,
    epochs=number_epochs,
    max_queue_size=batch_size * 2,
    verbose=1,
)

# %% Save model

model.save(config.INCEPTIONV3_MODEL_PATH)

# %% Show graphics

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, number_epochs), H1.history["loss"], label="train_loss")
plt.plot(np.arange(0, number_epochs), H1.history["val_loss"], label="val_loss")
plt.plot(
    np.arange(0, number_epochs),
    H1.history["accuracy"],
    label="train_acc",
)
plt.plot(
    np.arange(0, number_epochs),
    H1.history["val_accuracy"],
    label="val_acc",
)
plt.title("Pérdida de entrenamiento y accuracy de la primera fase")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
