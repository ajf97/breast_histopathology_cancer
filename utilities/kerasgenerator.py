# -*- coding: utf-8 -*-
__author__ = "Alejandro Jerónimo Fuentes"
__license__ = "GPL"
__date__ = "19/08/2020"
__status__ = "Development"


# Load packages
from keras.utils import np_utils
import numpy as np
import h5py


class KerasGenerator:

    def __init__(self, db_path, batch_size, preprocessors=[]):

        self.preprocessors = preprocessors
        self.batch_size = batch_size
        self.db = h5py.File(db_path, "r")
        self.num_images = self.db["images"].shape[0]

    def generate_image_batch(self):

        while True:
            for i in np.arange(0, self.num_images, self.batch_size):

                images = self.db["images"][i: i + self.batch_size]
                labels = self.db["labels"][i: i + self.batch_size]

                # Apply preprocessors to images on batch

                if [] not in self.preprocessors:
                    images_preprocessed = []

                    for image in images:
                        for preprocessor in self.preprocessors:
                            image = preprocessor.preprocess(image)

                        images_preprocessed.append(image)

                    images = np.array(images_preprocessed)

                # Categorizing labels

                labels_categorized = []

                for label in labels:
                    label = np_utils.to_categorical(label, num_classes=2)
                    labels_categorized.append(label)

                labels = np.array(labels_categorized)

                yield (images, labels)

    def close_db(self):
        self.db.close()
