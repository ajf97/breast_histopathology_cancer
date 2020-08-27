# -*- coding: utf-8 -*-
__author__ = "Alejandro Jerónimo Fuentes"
__date__ = "21/08/2020"

# %% Import neccesary packages

from keras.layers.core import Dense, Dropout, Flatten

# %% Layer definition


class FCLayer:
    @staticmethod
    def build(baseModel, numNeurons):

        # Add a FC layer
        model = baseModel.output
        model = Flatten(name="flatten")(model)
        model = Dense(numNeurons, activation="relu")(model)
        model = Dropout(0.5)(model)
        model = Dense(2, activation="softmax")(model)

        return model
