# -*- coding: utf-8 -*-
__author__ = "Alejandro Jerónimo Fuentes"
__date__ = "13/08/2020"

# %% Importar paquetes necesarios
from config import breast_histopathology_cancer_config as config
from utilities.hdf5dataset import HDF5Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from imblearn.datasets import make_imbalance
import numpy as np
import json
import cv2
import os

# %% Obtener rutas de las imágenes y sus etiquetas correspondientes

trainPaths = []
trainLabels = []

for root, dirs, files in os.walk(config.IMAGE_PATH):
    if files is not []:
        for imageFile in files:

            filename = os.path.join(root, imageFile)
            label = root.split(os.sep)[-1]

            trainPaths.append(filename)
            trainLabels.append(label)

# %% Normalizar etiquetas

le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# %% Balancear clases utilizando under-sampling

trainPaths = np.array(trainPaths).reshape((-1, 1))
trainLabels = np.array(trainLabels)

X, y = make_imbalance(trainPaths, trainLabels,
                      sampling_strategy={0: 10000, 1: 10000},
                      random_state=42)
X = X.flatten()


# %% Separar las rutas en conjuntos de entrenamiento, prueba y validación

split = train_test_split(X, y,
                         test_size=config.NUM_TEST_IMAGES,
                         stratify=y)

(trainPaths, testPaths, trainLabels, testLabels) = split

# A continuación, obtenemos el conjunto de validación

split = train_test_split(trainPaths, trainLabels,
                         test_size=config.NUM_VAL_IMAGES,
                         stratify=trainLabels)

(trainPaths, valPaths, trainLabels, valLabels) = split

# %% Construcción del dataset

dataset = [
    ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
    ("test", testPaths, testLabels, config.TEST_HDF5),
    ("val", valPaths, valLabels, config.VAL_HDF5)
    ]

(R, G, B) = ([], [], [])


for (dtype, paths, labels, hdf5_path) in dataset:

    # Crear base de datos hdf5
    hdf5 = HDF5Dataset((len(paths), 50, 50, 3), hdf5_path)

    # Mostramos información de progreso por pantalla
    print("[INFO] Creando la base de datos " + dtype)

    for (path, label) in zip(paths, labels):

        # Lectura de la imagen con OpenCV
        image = cv2.imread(path)

        # Calculamos la media RGB de la imagen
        if dtype == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        # Convertimos la imagen a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        image = np.expand_dims(image, axis=0)


        # Escribimos la imagen en la base de datos hdf5
        hdf5.add([image], [label])

    hdf5.close()

# %% Serializamos la normalización RGB

print("[INFO] Normalización RGB")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.MEAN_PATH, "w")
f.write(json.dumps(D))
f.close()

# %% Calculamos los pesos de las clases para corregir el desbalanceo

# class_weights = class_weight.compute_class_weight(class_weight='balanced',
#                                                   classes=np.unique(trainLabels),
#                                                   y=trainLabels)


# print("[INFO] Calculando pesos de las clases")
# D = {"0": class_weights[0], "1": class_weights[1]}
# f = open(config.CLASS_WEIGHTS_PATH, "w")
# f.write(json.dumps(D))
# f.close()

















