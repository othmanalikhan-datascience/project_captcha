"""
Module that is responsible for handling raw data.
"""

import glob
import pathlib
import os
import random

import cv2
import imutils


class ImageHandler:
    """
    Responsible for handling image reading and writing.
    """

    def __init__(self, PATH_DATA):
        """
        :param PATH_DATA: String, path to the root directory containing data.
        """
        self.PATH_DATA = PATH_DATA

    def read(self, imageNum, PATH_SEARCH_DIR):
        """
        Reads an image in the given directory based on its image number.

        The convention is that each image is named as <NUM>_<CAPTCHA>.<ext>.
        For instance, 000395_XL3H.jpg

        :param imageNum: Integer, the number of the image.
        :param PATH_SEARCH_DIR: String, path relative to the 'data' directory.
        :return: 2-Tuple, (imageData, imageName, imageNumberAsString)
        """
        PATH_SEARCH = os.path.join(self.PATH_DATA, PATH_SEARCH_DIR)
        SEARCH_STRING = '%06d' % imageNum
        PATH_IMAGE = glob.glob(f"{PATH_SEARCH}/{SEARCH_STRING}*")

        if PATH_IMAGE:
            PATH_IMAGE = PATH_IMAGE[0]
            img = cv2.imread(PATH_IMAGE, 1)
            fName = pathlib.PurePath(PATH_IMAGE).name.split(".")[0]
            index, label = fName.split("_")
            return img, label, index
        else:
            raise KeyError(f"Could not find image number '{SEARCH_STRING}' in '{PATH_SEARCH}'")

    def write(self, image, imageName, imageNum, PATH_DIR):
        """
        Writes the given image to the appropriate directory type with the
        given frame number.

        :param img: cv2.Image, containing the letter.
        :param imageName: String, the name of the image.
        :param imageNum: String, the number of the image.
        :param PATH_DIR: String, path to the directory to store letters.
        """
        PATH_DIR = os.path.join(self.PATH_DATA, PATH_DIR)
        os.makedirs(PATH_DIR, exist_ok=True)
        PATH_IMAGE = os.path.join(PATH_DIR, f"{imageNum}_{imageName}.jpg")
        cv2.imwrite(PATH_IMAGE, image)

    def writeLetter(self, img, label, PATH_DIR):
        """
        Writes the given image to the appropriate directory type with the
        given frame number.

        :param img: cv2.Image, containing the letter.
        :param label: String, the label of the letter (e.g. "Z")
        :param PATH_DIR: String, path to the directory to store letters.
        """
        # Create dirs as need be
        PATH_DIR = os.path.join(self.PATH_DATA, PATH_DIR, label)
        os.makedirs(PATH_DIR, exist_ok=True)

        # Check if files exist. If so, save image as lastFileStored+1.jpg
        fs = glob.glob(f"{PATH_DIR}/*")
        fPath = os.path.join(PATH_DIR, f"{len(fs)+1}.jpg")
        cv2.imwrite(fPath, img)


def resizeToFit(image, width, height):
    """
    Resize an image to fit within a given size.

    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """

    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    return image

