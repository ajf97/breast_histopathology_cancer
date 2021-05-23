# -*- coding: utf-8 -*-
__author__ = "Alejandro Jerónimo Fuentes"
__date__ = "21/08/2020"


# %% Import neccesary packages

import os

import cv2

import config.breast_histopathology_cancer_config as config

# %% Remove images with different shape to 50x50

for root, dirs, files in os.walk(config.IMAGE_PATH):
    if files is not []:
        for imageFile in files:
            file = os.path.join(root, imageFile)
            image = cv2.imread(file)

            if image.shape[0] != 50 or image.shape[1] != 50:
                os.remove(file)
