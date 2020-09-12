# -*- coding: utf-8 -*-
__author__ = "Alejandro Jerónimo Fuentes"
__date__ = "20/08/2020"

# %% Import neccesary packages

import config.breast_histopathology_cancer_config as config
from matplotlib import pyplot as plt
from models.alexnet import AlexNet
from utilities import preprocessing
from utilities.kerasgenerator import KerasGenerator
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import json


# %% Load values RGB

values_rgb = json.loads(open(config.MEAN_PATH).read())


# %% Initialize preprocessors and keras generators

batch_size = 32
number_epochs = 15

aug = ImageDataGenerator(horizontal_flip=True,
                         vertical_flip=True,
                         rotation_range=90,
                         zoom_range=[0.5, 1.0]
                         )


rp = preprocessing.ResizePreprocessor(256, 256)
cp = preprocessing.RandomCropPreprocessor(227, 227)
mrgb = preprocessing.MeanRGBPreprocessor(values_rgb)

kgtrain = KerasGenerator(config.TRAIN_HDF5, batch_size,
                         preprocessors=[rp, cp, mrgb],
                         data_augmentation=aug)

rp = preprocessing.ResizePreprocessor(227, 227)

kgval = KerasGenerator(config.VAL_HDF5, batch_size, preprocessors=[rp, mrgb],
                       data_augmentation=aug)

# %% Compiling model

print("[INFO] Compiling AlexNet")
sgd = SGD(lr=0.001)
model = AlexNet.build()
model.compile(loss="categorical_crossentropy", optimizer=sgd,
              metrics=["accuracy"])

# %% Training model

print("[INFO] Training AlexNet")
H = model.fit(kgtrain.generate_image_batch(),
              steps_per_epoch=kgtrain.num_images // batch_size,
              validation_data=kgval.generate_image_batch(),
              validation_steps=kgval.num_images // batch_size,
              epochs=number_epochs,
              max_queue_size=batch_size * 2,
              verbose=1
              )

# %% Save model

model.save(config.ALEXNET_MODEL_PATH, overwrite=True)
kgtrain.close_db()
kgval.close_db()

# %% Show graphics

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, number_epochs), H.history["loss"],
         label="train_loss")
plt.plot(np.arange(0, number_epochs), H.history["val_loss"],
         label="val_loss")
plt.plot(np.arange(0, number_epochs), H.history["accuracy"],
         label="train_acc")
plt.plot(np.arange(0, number_epochs), H.history["val_accuracy"],
         label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
