import glob
import os
import pickle

import cv2
import numpy as np
from keras.models import load_model

from data import ImageHandler
from data import resizeToFit
from filter import ImageFilter


GREEN = (0, 255, 0)


class Solver:

    def solveLetter(self, img, model, labeller):
        """
        Predicts the letter contained in the image by feeding the image
        into a neural network.

        :param img: cv2.Image, the image of the letter.
        :param model: keras.engine.sequential.Sequential, the neural network model.
        :param labeller: sklearn.preprocessing.label.LabelBinarizer, contains labels.
        :return: Character, the predicted letter in the image.
        """
        # Re-size the letter image to 20x20 pixels to match training data
        # Turn the single image into a 4d list of images to make Keras happy
        img = resizeToFit(img, 20, 20)
        img = np.expand_dims(img, axis=2)
        img = np.expand_dims(img, axis=0)

        # Ask the neural network to make a prediction
        # Convert the one-hot-encoded prediction back to a normal letter
        prediction = model.predict(img)
        img = labeller.inverse_transform(prediction)[0]
        return img

    def solveCaptcha(self, img, model, labeller, imageFilter):
        """
        Attempts to solve the captcha via a neural network.

        :param img: cv2.Image, the image of the captcha.
        :param model: keras.engine.sequential.Sequential, the neural network model.
        :param labeller: sklearn.preprocessing.label.LabelBinarizer, contains labels.
        :return: 2-Tuple, (solvedCaptchaAsString, solvedCaptchaAsImage).
        """
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        letterRegions = imageFilter.computeLetterDetectionAlgorithm(img)

        # Preparing to draw output on an image
        predictions = []
        OFFSET = 20
        outImage = cv2.copyMakeBorder(img, OFFSET, 0, 0, 0, cv2.BORDER_CONSTANT)

        # Solving for each letter in captcha
        for (x0, y0, x1, y1) in letterRegions:
            letter = self.solveLetter(grayscale[y0:y1, x0:x1], model, labeller)
            predictions.append(letter)

            # draw the prediction on the output image
            cv2.rectangle(outImage, (x0, y0+OFFSET), (x1, y1+OFFSET), GREEN, 1)
            cv2.putText(outImage, letter, (x0, y0+15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 1)

        captcha = ''.join(predictions)
        return captcha, outImage

    def analyseResults(self, results):
        """
        Analyses the results of the neural network to give meaningful metrics.

        :param results: Dictionary, mapping expected value to predicted.
        """
        total = len(results)
        correct = 0

        for expected, predicted in results.items():
            if expected == predicted:
                correct += 1
            else:
                print(f"Misclassification: {predicted} != {expected}")

        print("Total: ", total)
        print("Incorrect: ", total - correct)
        print("Accuracy: ", correct/total * 100)

    def run(self, PATH_DATA, PATH_MODEL, PATH_LABEL):
        """
        Runs the solver against the given data directory

        :param PATH_DATA: String, path to the data containing letter images.
        :param PATH_MODEL: String, path to the output model file.
        :param PATH_LABEL: String, path to the output labels file.
        """
        results = {}
        imageFilter = ImageFilter()
        imageHandler = ImageHandler(os.path.join(PATH_DATA, ".."))
        numberImages = len(glob.glob(f"{PATH_DATA}/*"))

        # Loading trained model
        with open(PATH_LABEL, "rb") as f:
            labeller = pickle.load(f)
        model = load_model(PATH_MODEL)

        # Solve captchas and save output
        for i in range(1, numberImages+1):
            img, solution, num = imageHandler.read(i, "input")
            captcha, outImage = self.solveCaptcha(img, model, labeller, imageFilter)
            results[solution] = captcha

            if captcha == solution:
                classify = "output/solved_correct"
            else:
                classify = "output/solved_incorrect"
            imageHandler.write(outImage, solution, num, classify)
            imageHandler.write(outImage, solution, num, "output/solved")

        self.analyseResults(results)
