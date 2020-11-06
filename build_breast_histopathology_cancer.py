# -*- coding: utf-8 -*-
__author__ = "Alejandro Jerónimo Fuentes"
__date__ = "13/08/2020"

# %% Import neccesary packages
from config import breast_histopathology_cancer_config as config
from utilities.hdf5dataset import HDF5Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.datasets import make_imbalance
import numpy as np
import pandas as pd
import json
import cv2
import os

# %% Get images and labels path

trainPaths = []
trainLabels = []

# Dataframe definition

df = pd.DataFrame(columns=["patient_id", "x", "y", "target", "path"])
i = 1

for root, dirs, files in os.walk(config.IMAGE_PATH):
    if files is not []:
        for imageFile in files:

            filename_path = os.path.join(root, imageFile)
            label = root.split(os.sep)[-1]

            # Parsing files
            filename = filename_path.split("\\")[-1]
            filename = filename.split("_")

            patient_id = filename[0]
            coord_x = int(filename[2][1:])
            coord_y = int(filename[3][1:])

            row = [patient_id, coord_x, coord_y, label, filename_path]

            trainPaths.append(filename_path)
            trainLabels.append(label)
            df.loc[len(df)] = row

df.to_csv(config.DATAFRAME_PATH, index=False)

# %% Label encoding

le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# %% Balancing classes with under-sampling

trainPaths = np.array(trainPaths).reshape((-1, 1))
trainLabels = np.array(trainLabels)

X, y = make_imbalance(trainPaths, trainLabels,
                      sampling_strategy={0: 10000, 1: 10000},
                      random_state=42)
X = X.flatten()


# %% Split data

split = train_test_split(X, y,
                         test_size=config.NUM_TEST_IMAGES,
                         stratify=y)

(trainPaths, testPaths, trainLabels, testLabels) = split

# Split validation set

split = train_test_split(trainPaths, trainLabels,
                         test_size=config.NUM_VAL_IMAGES,
                         stratify=trainLabels,
                         random_state=42)

(trainPaths, valPaths, trainLabels, valLabels) = split

# %% Dataset construction

dataset = [
    ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
    ("test", testPaths, testLabels, config.TEST_HDF5),
    ("val", valPaths, valLabels, config.VAL_HDF5)
    ]

(R, G, B) = ([], [], [])


for (dtype, paths, labels, hdf5_path) in dataset:

    # Create HDF5 database
    hdf5 = HDF5Dataset((len(paths), 50, 50, 3), hdf5_path)

    print("[INFO] Creando la base de datos " + dtype)

    for (path, label) in zip(paths, labels):

        # Read image with OpenCV
        image = cv2.imread(path)

        # Mean RGB subtraction
        if dtype == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        # Convert image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        image = np.expand_dims(image, axis=0)

        # Write image into hdf5 dataset
        hdf5.add([image], [label])

    hdf5.close()

# %% Serialize mean

print("[INFO] Normalización RGB")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.MEAN_PATH, "w")
f.write(json.dumps(D))
f.close()
