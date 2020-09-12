# -*- coding: utf-8 -*-
__author__ = "Alejandro Jerónimo Fuentes"
__date__ = "25/08/2020"


# %% Import neccesary packages

import config.breast_histopathology_cancer_config as config
from sklearn.metrics import classification_report
from utilities.kerasgenerator import KerasGenerator
from utilities.metrics import rank1_accuracy
from keras.models import load_model
from keras.utils import np_utils
from scikitplot.metrics import plot_confusion_matrix
import utilities.preprocessing as preprocessors
import h5py
import json


# %% Loading model

#db = h5py.File("C:\\Users\\ajf97\\Documents\\TFG\\large_dataset_hdf5\\train.hdf5", "r")
db = h5py.File(config.TEST_HDF5, "r")
model = load_model(config.VGG19_MODEL_PATH)
values_rgb = json.loads(open(config.MEAN_PATH).read())


# %% Define preprocessors and generator to use

batch_size = 32

rp = preprocessors.ResizePreprocessor(224, 224)
# mzo = preprocessors.MeanZeroOnePreprocessor()
mrgb = preprocessors.MeanRGBPreprocessor(values_rgb)
#kg = KerasGenerator("C:\\Users\\ajf97\\Documents\\TFG\\large_dataset_hdf5\\train.hdf5", batch_size, preprocessors=[rp, mrgb])
kg = KerasGenerator(config.TEST_HDF5, batch_size, preprocessors=[rp, mrgb])

# %% Evaluate model

predictions = model.predict_generator(kg.generate_image_batch(),
                                      steps=kg.num_images / batch_size,
                                      max_queue_size=batch_size * 2,
                                      verbose=1)


# %% Print classification report

testY = np_utils.to_categorical(db["labels"], num_classes=2)

print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=["non-cancer", "cancer"]))

# %% Print rank-1 metric

rank1 = rank1_accuracy(predictions, db["labels"])
print("Rank-1: ", rank1)

# %% Plot confussioin matrix

plot_confusion_matrix(testY.argmax(axis=1),
                      y_pred=predictions.argmax(axis=1))

db.close()
