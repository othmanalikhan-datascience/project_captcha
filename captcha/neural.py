"""
Contains the logic of the Neural Network.
"""
import glob
import pathlib
import pickle

import cv2
import numpy as np
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from data import resizeToFit


class NeuralNetwork:

    def __init__(self, PATH_DATA, PATH_MODEL, PATH_LABEL):
        """
        :param PATH_DATA: String, path to the data containing letter images.
        :param PATH_MODEL: String, path to the output model file.
        :param PATH_LABEL: String, path to the output labels file.
        """
        self.PATH_DATA = PATH_DATA
        self.PATH_MODEL = PATH_MODEL
        self.PATH_LABEL = PATH_LABEL
        self.model = None

    def loadData(self):
        """
        Loads the images of individual letters from the given directory.
        """
        data = []
        labels = []

        for letter in glob.glob(f"{self.PATH_DATA}/*/*.jpg"):
            img = cv2.imread(letter, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = resizeToFit(img, 20, 20)
            img = np.expand_dims(img, axis=2)   # 3rd channel to make Keras happy
            label = pathlib.PurePath(letter).parent.name

            data.append(img)
            labels.append(label)

        # scale the raw pixel intensities to the range [0, 1] (this improves training)
        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)
        return data, labels

    def build(self):
        """
        Builds a neural network for analysing captcha images.
        """
        model = Sequential()

        # First convolutional layer with max pooling
        model.add(Conv2D(20, (5, 5), padding="valid", input_shape=(20, 20, 1), activation="relu"))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        # Hidden layer with 2000 nodes
        model.add(Flatten())
        model.add(Dense(2000, activation="relu"))

        # Output layer with 36 nodes (one for each possible letter/number we predict)
        model.add(Dense(36, activation="softmax"))

        # Ask Keras to build the TensorFlow model behind the scenes
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        self.model = model

    def train(self):
        """
        Trains the neural network.
        """
        data, labels = self.loadData()
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, labels, test_size=0.25, random_state=0)

        # Convert the labels (letters) into one-hot encodings that Keras can work with
        lb = LabelBinarizer().fit(Ytrain)
        Ytrain = lb.transform(Ytrain)
        Ytest = lb.transform(Ytest)

        # Save the mapping from labels to one-hot encodings.
        # We'll need this later when we use the model to decode what it's predictions mean
        with open(self.PATH_LABEL, "wb") as f:
            pickle.dump(lb, f)

        # train the neural network
        history = self.model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), batch_size=32, epochs=5, verbose=1)
        self.model.save(self.PATH_MODEL)

        # import matplotlib.pyplot as plt
        # Plot training & validation accuracy values
        # plt.plot(history.history['acc'])
        # plt.plot(history.history['val_acc'])
        # plt.title('Model accuracy')
        # plt.ylabel('Accuracy')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()

        # Plot training & validation loss values
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('Model loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()

