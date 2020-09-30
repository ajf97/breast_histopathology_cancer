# -*- coding: utf-8 -*-
__author__ = "Alejandro Jerónimo Fuentes"
__date__ = "29/09/2020"

# Rutas de las imágenes de entrenamiento
IMAGE_TRAIN_FULL_PATH = "D://dataset//CBIS-DDSM-TRAIN-FULL//"
IMAGE_TRAIN_ROI_PATH = "D://dataset//CBIS-DDSM-TRAIN-ROI//"

# Rutas de las imágenes de prueba
IMAGE_TEST_FULL_PATH = "D://dataset//CBIS-DDSM-TEST-FULL//"
IMAGE_TEST_ROI_PATH = "D://dataset//CBIS-DDSM-TEST-ROI//"

NUM_CLASSES = 2
NUM_VAL_IMAGES = 50
NUM_TEST_IMAGES = 50
NUM_TRAIN_IMAGES = 50

# Rutas donde se van a almacenar los archivos HDF5
TRAIN_HDF5 = "D://dataset//hdf5//train.hdf5"
TEST_HDF5 = "D://dataset//hdf5//test.hdf5"
VAL_HDF5 = "D://dataset//hdf5//val.hdf5"

# Dataframes
TRAIN_DATAFRAME = "D://dataset//calc_case_description_train_set.csv"
TEST_DATAFRAME = "D://dataset//calc_case_description_test_set.csv"

# Resolución
INPUT_SHAPE = 572
