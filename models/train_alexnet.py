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
import numpy as np
import json

# %% Get class weights

# class_weights = json.loads(open(config.CLASS_WEIGHTS_PATH).read())
# class_weights = dict(enumerate([class_weights["0"], class_weights["1"]]))

# %% Initialize preprocessors and keras generators

batch_size = 64
number_epochs = 10

rp = preprocessing.ResizePreprocessor(256, 256)
cp = preprocessing.RandomCropPreprocessor(227, 227)
mzop = preprocessing.MeanZeroOnePreprocessor()

kgTrain = KerasGenerator(config.TRAIN_HDF5, batch_size, preprocessors=[rp, cp, mzop])

rp = preprocessing.ResizePreprocessor(227, 227)

kgVal = KerasGenerator(config.VAL_HDF5, batch_size, preprocessors=[rp, mzop])

# %% Compiling model

print("[INFO] Compiling AlexNet")
sgd = SGD(lr=0.001)
model = AlexNet.build()
model.compile(loss="categorical_crossentropy", optimizer=sgd,
              metrics=["accuracy"])
#model.load_weights(config.ALEXNET_WEIGHTS_PATH)

# %% Training model

print("[INFO] Training AlexNet")
model.fit(kgTrain.generate_image_batch(),
          steps_per_epoch=kgTrain.num_images // batch_size,
          validation_data=kgVal.generate_image_batch(),
          validation_steps=kgVal.num_images // batch_size,
          epochs=number_epochs,
          max_queue_size=batch_size * 2,
          verbose=1
          )

# %% Save model

model.save(config.ALEXNET_MODEL_PATH, overwrite=True)
kgTrain.close_db()
kgVal.close_db()

# %% Show graphics

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, number_epochs), model.history["loss"],
         label="train_loss")
plt.plot(np.arange(0, number_epochs), model.history["val_loss"],
         label="val_loss")
plt.plot(np.arange(0, number_epochs), model.history["accuracy"],
         label="train_acc")
plt.plot(np.arange(0, number_epochs), model.history["val_accuracy"],
         label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
