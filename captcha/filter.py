"""
Module that contains the computer vision algorithms.
"""


import cv2
import time
import numpy as np


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (127, 127, 127)
RED = (255, 0, 0)


def printStatus(algorithName):
    """
    Decorator that wraps a function with print statements that indicate
    process status of the function.

    :param algorithName: String, the name used in the print statements.
    :return: Function, the wrapped function.
    """
    def decorator(func):
        def wrapper(*args):
            print(f">>> PRE-RENDERING {str.upper(algorithName)} IMAGES STARTING <<<")
            func(*args)
            print(f">>> PRE-RENDERING {str.upper(algorithName)}IMAGES COMPLETE <<<")
        return wrapper
    return decorator


class ImageFilter:
    """
    Responsible for applying filtering algorithms on the images.
    """

    def __init__(self):
        """
        A simple constructor that initializes variables used to crudely
        time algorithms.
        """
        self.currentFrame = 0
        self.timeStart = None

    def computeOtsuAlgorithm(self, img, *args):
        """
        Wrapper function that simply runs the OpenCV Otsu's thresholding.

        :param img: The image to perform the algorithm on.
        :return: The outputs of the OpenCV thresholding function.
        """
        return cv2.threshold(img, *args)

    def computeDifferenceAlgorithm(self, img1, img2):
        """
        Computes the absolute difference between two frames.

        :param img1: The first image to be used in the difference.
        :param img2: The second image to be used in the difference.
        :return: The difference of two images in OpenCV format.
        """
        return cv2.absdiff(img1, img2)

    def computeLetterDetectionAlgorithm(self, img):
        """
        Converts the image to grayscale, applies a binary threshold, dilates
        the image, then finds contours. Afterwards, draws contours with
        sufficient area size onto the image.

        :param img: An image.
        :return: An image resulting from applying the algorithm on the input.
        :return: List, containing coordinates of individual letters.
        """
        # Utility function to split regions
        def splitRegion(x, y, w, h, num):
            regions = []
            for i in range(1, num+1):
                fraction = w // num
                regions.append((x + (i-1)*fraction, y, x + i*fraction, y + h))
            return regions

        # Letter in captcha = 13 tall * N wide (pixels)
        contourMinArea = 13 * 4
        contourMaxArea = 13 * 13 * 4        # 2x when letters joined

        # Converts the image to grayscale
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Applies global thresholding to binary
        args = (grayscale, 127, 255, cv2.THRESH_BINARY)
        _, thresh = self.computeOtsuAlgorithm(*args)

        # Fill in around the border of the image. This is so that the contour
        # algorithm doesn't think individual letter contours are merged with
        # the global image contour. e.g. The bottom of 'J' merges with the
        # rest of the image if there are black pixels at the bottom touching 'J'
        thresh[0] = np.array(60*[255])
        thresh[19] = np.array(60*[255])
        thresh[:, 0] = np.array(20*[255])
        thresh[:, 59] = np.array(20*[255])

        # Draws all contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # track = cv2.drawContours(img, contours, -1, RED)

        # Extracting only sufficiently large contours and drawing a
        # rectangle around them
        letterRegions = []
        for c in contours:
            if contourMaxArea >= cv2.contourArea(c) >= contourMinArea:
                (x, y, w, h) = cv2.boundingRect(c)
                # print("Contours Rectangle at: (%d %d) (%d %d)" % (x, y, w, h))
                # print("Contours Area: %d " % cv2.contourArea(c))

                # Compare the width and height of the contour to detect letters that
                # are conjoined into one chunk
                if w / h > 1.25:
                    letterRegions += splitRegion(x, y, w, h, 2)
                elif w / h > 2.25:
                    letterRegions += splitRegion(x, y, w, h, 3)
                elif w / h > 3.25:
                    letterRegions += splitRegion(x, y, w, h, 4)
                else:
                    letterRegions.append((x, y, x+w, y+h))

        # If too many regions are found, only select the first four (even
        # though they might be wrong)
        if len(letterRegions) > 4:
            letterRegions = letterRegions[:4]

        # If not enough letters found, resort to manually slicing the image.
        # Through trail and error, the numbers below work best for most
        # generated captchas
        elif len(letterRegions) < 4:
            letterRegions = [(5, 3, 18, 17), (18, 3, 30, 17),
                             (30, 3, 42, 17), (42, 3, 55, 17)]

        # Sort the detected letter images based on the x coordinate to make sure
        # we are processing them from left-to-right so we match the right image
        # with the right letter
        letterRegions = sorted(letterRegions)
        return letterRegions

    def _updatePerformanceMeasuring(self):
        """
        Updates the last frame called in the animation and prints the time
        elapsed, and average FPS since the last call to _beginTiming function.
        """
        print("--- PERFORMANCE MEASUREMENTS UPDATING ---")
        self.currentFrame += 1
        dt = time.time() - self.timeStart
        fps = (self.currentFrame / dt)

        print("Time Elapsed: %d seconds" % dt)
        print("Current Frame: %d" % self.currentFrame)
        print("Overall FPS: %.2f" % fps)

    def _startPerformanceMeasuring(self):
        """
        Starts the measuring of time and current frame.
        To be used in conjunction with it's update method.
        """
        print("--- PERFORMANCE MEASUREMENTS STARTING NOW ---")
        self.timeStart = time.time()
        self.currentFrame = 0


class AnimationPreRenderer:
    """
    Responsible for pre-rendering the output of the image filtering
    algorithms (otherwise real-time rendering is too slow).
    """

    def __init__(self, imageHandler):
        """
        A simple constructor.

        :param imageHandler: An instantiated ImageHandler object that is
        responsible for reading and writing to the correct directories.
        """
        self.imageHandler = imageHandler
        self.imageFilter = ImageFilter()

    @printStatus("Letter Tracking")
    def generateLetterDetectionImages(self, fStart, fEnd):
        """
        :param fStart: The number of first frame.
        :param fEnd: The number of the last frame.

        Generates and saves the images that show worm tracking algorithm.
        """
        for f in range(fStart, fEnd+1):
            print("Letter tracking rendering progress: %d/%d frames" % (f, fEnd))
            img, label, num = self.imageHandler.read(f, "input")
            letterRegions = self.imageFilter.computeLetterDetectionAlgorithm(img)

            # First save letters as individual images
            for i, (x0, y0, x1, y1) in enumerate(letterRegions):
                letterImage = img[y0:y1, x0:x1]
                letterLabel = label[i]
                self.imageHandler.writeLetter(letterImage, letterLabel, "output/letters")

            # Then save entire image with overlays on a black/white image
            grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            args = (grayscale, 127, 255, cv2.THRESH_BINARY)
            _, thresh = self.imageFilter.computeOtsuAlgorithm(*args)
            for i, (x0, y0, x1, y1) in enumerate(letterRegions):
                overlay = cv2.rectangle(thresh, (x0, y0), (x1, y1), BLACK, 1)
            self.imageHandler.write(overlay, num, label, "output/detection")

    @printStatus("Difference")
    def generateDifferenceImages(self, fStart, fEnd, fDiff):
        """
        :param fStart: The number of first frame.
        :param fEnd: The number of the last frame.
        :param fDiff: The difference range between two consecutive frames.

        Generates and saves the images that show the absolute difference
        between consecutive images.
        """
        for f in range(fStart, fEnd):
            print("Difference rendering progress: %d/%d frames" % (f, fEnd))
            img1, num, label = self.imageHandler.read(f, "input")
            img2, _, _ = self.imageHandler.read(f + fDiff, "input")
            diff = self.imageFilter.computeDifferenceAlgorithm(img1, img2)
            self.imageHandler.write(diff, num, label, "output/difference")

    @printStatus("Otsu")
    def generateOtsuImages(self, fStart, fEnd):
        """
        Generates and saves the images that show Otsu's thresholding.

        :param fStart: The number of first f.
        :param fEnd: The number of the last f.
        """
        for f in range(fStart, fEnd+1):
            print("Otsu rendering progress: %d/%d frames" % (f, fEnd))
            img, num, label = self.imageHandler.read(f, "input")
            args = (img, 110, 255, cv2.THRESH_BINARY)
            _, thresh = self.imageFilter.computeOtsuAlgorithm(*args)
            self.imageHandler.write(thresh, num, label, "output/otsu")
