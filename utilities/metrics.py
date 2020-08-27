# -*- coding: utf-8 -*-
__author__ = "Alejandro Jerónimo Fuentes"
__date__ = "25/08/2020"

# Import packages

import numpy as np


def rank1_accuracy(predictions, labels):
    rank1 = 0

    for (prediction, true_label) in zip(predictions, labels):
        prediction = np.argsort(prediction)[::-1]

        if prediction[0] == true_label:
            rank1 += 1

    rank1 /= float(len(labels))
    return rank1
