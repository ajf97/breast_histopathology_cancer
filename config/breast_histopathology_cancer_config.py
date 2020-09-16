# -*- coding: utf-8 -*-
__author__ = "Alejandro Jerónimo Fuentes"
__date__ = "13/08/2020"

# Ruta de las imágenes del conjunto de datos
IMAGE_PATH = "C://Users//ajf97//Documents//TFG//dataset//data"

NUM_CLASSES = 2
NUM_VAL_IMAGES = 0.15
NUM_TEST_IMAGES = 0.15

# Rutas donde se van a almacenar los archivos HDF5
TRAIN_HDF5 = "C://Users//ajf97//Documents//TFG//dataset//hdf5//train.hdf5"
TEST_HDF5 = "C://Users//ajf97//Documents//TFG//dataset//hdf5//test.hdf5"
VAL_HDF5 = "C://Users//ajf97//Documents//TFG//dataset//hdf5//val.hdf5"


# Ruta para almacenar los ficheros de salida
OUTPUT_PATH = "C://Users//ajf97//Documents//TFG//breast_histopathology_cancer//output"

# Ruta del archivo de la media RGB del dataset
MEAN_PATH = OUTPUT_PATH + "//breast_histopathology_cancer_mean.json"

# Ruta del dataframe asociado al dataset completo
DATAFRAME_PATH = OUTPUT_PATH + "//dataframe.csv"

# Ruta de los pesos de la arquitectura AlexNet preentrenados en Imagenet
ALEXNET_MODEL_PATH = "C://Users//ajf97//Documents//TFG//breast_histopathology_cancer//output//alexnet_model.h5"

# Ruta de los modelos
VGG16_MODEL_PATH = OUTPUT_PATH + "//vgg16_model.h5"
VGG19_MODEL_PATH = OUTPUT_PATH + "//vgg19_model.h5"
