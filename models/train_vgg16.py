# -*- coding: utf-8 -*-
__author__ = "Alejandro Jerónimo Fuentes"
__date__ = "21/08/2020"


# %% Import neccesary packages

from keras.applications import VGG16
from keras.optimizers import SGD, RMSprop, Adam
from utilities.fclayer import FCLayer
from keras.layers import Input
from keras.models import Model
from utilities.kerasgenerator import KerasGenerator
import config.breast_histopathology_cancer_config as config
import utilities.preprocessing as preprocessors
import matplotlib.pylab as plt
import numpy as np
import json

# %% Get class weights

#class_weights = json.loads(open(config.CLASS_WEIGHTS_PATH).read())
#class_weights = dict(enumerate([class_weights["0"], class_weights["1"]]))

values_rgb = json.loads(open(config.MEAN_PATH).read())


# %% Define preprocessors and generators

batch_size = 32
number_epochs = 16

rp = preprocessors.ResizePreprocessor(224, 224)
#mzon = preprocessors.MeanZeroOnePreprocessor()
mrgb = preprocessors.MeanRGBPreprocessor(values_rgb)

trainGen = KerasGenerator(config.TRAIN_HDF5, batch_size,
                          preprocessors=[rp, mrgb])
valGen = KerasGenerator(config.VAL_HDF5, batch_size, preprocessors=[rp, mrgb])

# %% Define base model

baseModel = VGG16(weights="imagenet", include_top=False,
                  input_tensor=Input(shape=(224, 224, 3)))

head = FCLayer.build(baseModel, 256)

model = Model(inputs=baseModel.input, outputs=head)

# %% Freeze every layer in original VGG16 architecture

for layer in baseModel.layers:
    layer.trainable = False

# %% Compile the new model

opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# %% Training model

print("[INFO] Training Model")
H = model.fit(trainGen.generate_image_batch(),
          steps_per_epoch=trainGen.num_images // batch_size,
          validation_data=valGen.generate_image_batch(),
          validation_steps=valGen.num_images // batch_size,
          epochs=number_epochs,
          max_queue_size=batch_size * 2,
          verbose=1
          )

# %% Save model

model.save(config.VGG16_MODEL_PATH)

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
