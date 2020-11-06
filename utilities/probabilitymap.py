# -*- coding: utf-8 -*-
__author__ = "Alejandro Jerónimo Fuentes"
__date__ = "12/09/2020"


# %% Import neccesary packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import config.breast_histopathology_cancer_config as config
import utilities.preprocessing as preprocessor
from keras.models import load_model
from skimage import io
import json

# %% Load dataframe

df = pd.read_csv(config.DATAFRAME_PATH)


# %% Get patient_id data

def get_patient_df(patient_id):
    return df.loc[df["patient_id"] == patient_id, :]


# %% Generate function

def visualise_breast_tissue(patient_id, df = df, model = None, crop_dimension = [50,50]):

    plt.xticks([])
    plt.yticks([])
    plt.axis("off")

    rp = preprocessor.ResizePreprocessor(224, 224)
    mrgb = preprocessor.MeanRGBPreprocessor(values_rgb)


    # Get patient dataframe
    p_df = get_patient_df(patient_id)

    # Get the dimensions of the breast tissue image
    max_coord = np.max((p_df.x, p_df.y))

    grid = 255*np.ones(shape = (max_coord + crop_dimension[0], max_coord + crop_dimension[1], 3)).astype(np.uint8)
    mask = 255*np.ones(shape = (max_coord + crop_dimension[0], max_coord + crop_dimension[1], 3)).astype(np.uint8)

    # Replace array values with values of the image
    for x ,y ,target ,path in zip(p_df['x'], p_df['y'], p_df['target'], p_df['path']):
            img = io.imread(path)
            grid[y:y+crop_dimension[1],x:x+crop_dimension[0]] = img

            if model is not None:
                img_preprocessed = mrgb.preprocess(rp.preprocess(img))
                img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
                prediction = model.predict(img_preprocessed).argmax(axis=1)[0]

                if prediction != 0:
                    mask[y:y+crop_dimension[1],x:x+crop_dimension[0]] = [255, 127, 0]

            elif target != 0:
                # Set the path with the color if target is cancerous
                mask[y:y+crop_dimension[1],x:x+crop_dimension[0]] = [0, 255, 0]

    # Apply alpha compositing
    alpha = 0.68
    img = (mask * (1.0 - alpha) + grid * alpha).astype('uint8')
    io.imshow(img)

    return img

# %% Generate WSI
values_rgb = json.loads(open(config.MEAN_PATH).read())
model = load_model(config.RESNET50_MODEL_PATH)
patient_id = 14191
img = visualise_breast_tissue(patient_id, model = None)
plt.imshow(img)
