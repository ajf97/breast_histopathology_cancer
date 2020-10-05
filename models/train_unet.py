# -*- coding: utf-8 -*-
__author__ = "Alejandro Jerónimo Fuentes"
__date__ = "02/10/2020"

# %% Import neccesary packages

from config import cbis_ddsm_config as config
import numpy as np
from keras_unet.utils import plot_imgs, get_augmented, plot_segm_history
from keras_unet.models import custom_unet
from tensorflow.keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded
from models.unet import get_unet_512
from keras.callbacks import ModelCheckpoint
import h5py

# %% Read images

db_train = h5py.File(config.TRAIN_HDF5, "r")
db_val = h5py.File(config.VAL_HDF5, "r")

x_train, y_train, x_val, y_val = (db_train["images"][:]/255.0,
                                  db_train["masks"][:]/255.0,
                                  db_val["images"][:]/255.0,
                                  db_val["masks"][:]/255.0)


# %% Define generators

train_gen = get_augmented(
    x_train, y_train, batch_size=2,
    data_gen_args = dict(
        rotation_range=15.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=50,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant'
    ))


# %% Define model

input_shape = x_train[0].shape

model = custom_unet(
    input_shape,
    use_batch_norm=False,
    num_classes=1,
    filters=64,
    dropout=0.2,
    output_activation='sigmoid'
)


## %% Compile model

model.compile(
    optimizer=SGD(lr=0.01, momentum=0.99),
    loss='binary_crossentropy',
    metrics=[iou, iou_thresholded]
)

# %% Train model
model_filename = 'segm_model_v0.h5'
callback_checkpoint = ModelCheckpoint(
    model_filename,
    verbose=1,
    monitor='val_loss',
    save_best_only=True,
)

history = model.fit_generator(
    train_gen,
    steps_per_epoch=100,
    epochs=10,

    validation_data=(x_val, y_val),
    callbacks=[callback_checkpoint]
)
# %% Plot

plot_segm_history(history)

