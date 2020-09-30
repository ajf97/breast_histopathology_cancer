# -*- coding: utf-8 -*-
__author__ = "Alejandro Jerónimo Fuentes"
__date__ = "29/09/2020"

# %% Importar paquetes necesarios

from config import cbis_ddsm_config as config
from utilities.hdf5dataset import HDF5Dataset
import pandas as pd
import pydicom as dicom
import numpy as np
import cv2

# %% Leer dataframes de entrenamiento y prueba

df_train = pd.read_csv(config.TRAIN_DATAFRAME)
df_test = pd.read_csv(config.TEST_DATAFRAME)

# %% Seleccionar vista CC

df_train_cc = df_train.loc[df_train['image view'] == 'CC']
df_test_cc = df_test.loc[df_test['image view'] == 'CC']

# %% Barajar dataframes

df_train_cc = df_train_cc.sample(frac=1,
                                 random_state=45).reset_index(drop=True)
df_test_cc = df_test_cc.sample(frac=1, random_state=45).reset_index(drop=True)


# %% Funcion para convertir a escala de grises
def convert_to_gray_scale(image):
    image = image.astype("float")
    image = (np.maximum(image, 0) / image.max()) * 255.0
    image = np.uint8(image)
    image = cv2.resize(image, (config.INPUT_SHAPE,
                               config.INPUT_SHAPE))
    return image

# %% Función para leer las imágenes


def read_images(df, image_paths, mask_paths, num_images):
    images = []
    masks = []

    for i in range(num_images):
        image_path = df['image file path'].iloc[i]
        image_full_path = image_paths + image_path.replace('/', '//')

        mask_path = df['ROI mask file path'].iloc[i]
        mask_full_path = mask_paths + mask_path.replace('/', '//')

        # Arreglar nombre de archivos

        image_full_path = image_full_path.split("//")
        image_full_path[-1] = "1-1.dcm"
        image_full_path = "//".join(image_full_path)

        mask_full_path = mask_full_path.split("//")
        mask_full_path[-1] = "1-2.dcm"
        mask_full_path = "//".join(mask_full_path)

        # Lectura de las imágenes

        image = dicom.dcmread(image_full_path).pixel_array
        mask = dicom.dcmread(mask_full_path).pixel_array

        # Convertir las imágenes a escala de grises
        image = convert_to_gray_scale(image)
        mask = convert_to_gray_scale(mask)

        images.append(image)
        masks.append(mask)

    return (images, masks)


# %% Leer imágenes de entrenamiento

train_images, train_masks = read_images(df_train_cc,
                                        config.IMAGE_TRAIN_FULL_PATH,
                                        config.IMAGE_TRAIN_ROI_PATH,
                                        config.NUM_TRAIN_IMAGES)

# %% Leer imágenes de prueba y validación

df_test = df_test_cc[:config.NUM_TEST_IMAGES]
df_val = df_test_cc[config.NUM_TEST_IMAGES+1:]

test_images, test_masks = read_images(df_test, config.IMAGE_TEST_FULL_PATH,
                                      config.IMAGE_TEST_ROI_PATH,
                                      config.NUM_TEST_IMAGES)


val_images, val_masks = read_images(df_val, config.IMAGE_TEST_FULL_PATH,
                                    config.IMAGE_TEST_ROI_PATH,
                                    config.NUM_VAL_IMAGES)


# %% Convertir a arrays de Numpy

train_images = np.array(train_images)
train_masks = np.array(train_masks)

test_images = np.array(test_images)
test_masks = np.array(test_masks)

val_images = np.array(val_images)
val_masks = np.array(val_masks)

# %% Construcción del dataset

dataset = [
    ("train", train_images, train_masks, config.TRAIN_HDF5),
    ("test", test_images, test_masks, config.TEST_HDF5),
    ("val", val_images, val_masks, config.VAL_HDF5)
    ]


for (dtype, images, masks, hdf5_path) in dataset:

    # Crear base de datos hdf5
    hdf5 = HDF5Dataset((len(images), config.INPUT_SHAPE,
                        config.INPUT_SHAPE), hdf5_path, labelkey="masks",
                       masks=True)

    # Mostramos información de progreso por pantalla
    print("[INFO] Creando la base de datos " + dtype)

    for (image, mask) in zip(images, masks):
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Escribimos la imagen en la base de datos hdf5
        hdf5.add([image], [mask])

    hdf5.close()

