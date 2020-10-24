# -*- coding: utf-8 -*-
__author__ = "Alejandro Jerónimo Fuentes"
__date__ = "08/10/2020"


# %% Import neccesary packages

from config import tnbc_config as config
import numpy as np
import cv2
import os

# %% Read images and masks

images = []
masks = []

for root, dirs, files in os.walk(config.IMAGE_PATH):
    if files is not []:
        for imageFile in files:

            image_path = os.path.join(root, imageFile)
            mask_path = os.path.join(config.MASK_PATH, imageFile)

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path)
            mask = mask[:, :, 0]
            mask = np.expand_dims(mask, axis=2)

            images.append(image)
            masks.append(mask)


# %% Save images into a npy file

images = np.array(images)
masks = np.array(masks)

np.save(config.OUTPUT_IMAGES_PATH, images)
np.save(config.OUTPUT_MASKS_PATH, masks)
