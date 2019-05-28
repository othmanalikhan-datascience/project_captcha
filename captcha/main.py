"""
Main script for analysing captcha images and running a neural network on them.
"""

__date__ = "2019-05-26"
__author__ = "Othman Alikhan"

import os
import shutil

from controller import ImageController
from data import ImageHandler
from filter import AnimationPreRenderer
from neural import NeuralNetwork
from solver import Solver


def main():
    param = \
        {
            "fStart":       1,
            "fEnd":         1041,
            "fInterval":    500,  # milliseconds
            "fDiff":        1,
            "fSpeedFactor": 1,
        }

    PATH_DATA = os.path.join("..", "data")
    PATH_INP = os.path.join(PATH_DATA, "input")
    PATH_OUT = os.path.join(PATH_DATA, "output")

    PATH_MODEL = os.path.join(PATH_OUT, "model.hdf5")
    PATH_LABEL = os.path.join(PATH_OUT, "labels.dat")
    PATH_TRAIN = os.path.join(PATH_OUT, "letters")

    # Cleanup old results
    shutil.rmtree(PATH_OUT, ignore_errors=True)
    os.makedirs(PATH_OUT, exist_ok=True)

    # Pre-rendering images
    preRenderer = AnimationPreRenderer(ImageHandler(PATH_DATA))
    preRenderer.generateLetterDetectionImages(param["fStart"], param["fEnd"])
    preRenderer.generateOtsuImages(param["fStart"], param["fEnd"])
    # preRenderer.generateDifferenceImages(param["fStart"], param["fEnd"], param["fDiff"])
    # imageController = ImageController(param, PATH_DATA)
    # imageController.preRenderAllAnimation(param["fDiff"])

    # Train our neural network
    neuralNetwork = NeuralNetwork(PATH_TRAIN, PATH_MODEL, PATH_LABEL)
    neuralNetwork.build()
    neuralNetwork.train()

    # Solve for our data
    solver = Solver()
    solver.run(PATH_INP, PATH_MODEL, PATH_LABEL)

    # Choose display mode
    imageController = ImageController(param, PATH_DATA)
    imageController.runInteractiveMode()
    imageController.runAnimationMode(param["fInterval"], param["fSpeedFactor"])


if __name__ == "__main__":
    main()
