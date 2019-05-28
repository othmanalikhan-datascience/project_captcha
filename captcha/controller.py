"""
Controller class from the Model-View-Controller (MVC) model.
"""

import sys

from pyqtgraph import QtGui

from data import ImageHandler
from filter import AnimationPreRenderer
from gui import ImageDisplay


class ImageController:
    """
    Responsible for coordinating the GUI with the core image processing
    algorithms.
    """

    def __init__(self, args, PATH_DATA):
        """
        :param args: dictionary, containing the control parameters.
        :param PATH_DATA: String, path to the root data directory.
        """
        self.fStart = args["fStart"]
        self.fEnd = args["fEnd"]
        self.imageHandler = ImageHandler(PATH_DATA)
        self.initializeGUI()

    def initializeGUI(self):
        """
        Initializes the Qt GUI framework and application.
        """
        self.app = QtGui.QApplication(sys.argv)
        self.display = ImageDisplay(self.fStart, self.fEnd, self.imageHandler)

    def preRenderAllAnimation(self, fDiff):
        """
        Generates images for all possible animations.

        :param fDiff: The difference range between two consecutive frames.
        """
        preRenderer = AnimationPreRenderer(self.imageHandler)
        preRenderer.generateOtsuImages(self.fStart, self.fEnd)
        preRenderer.generateDifferenceImages(self.fStart, self.fEnd, fDiff)
        preRenderer.generateLetterDetectionImages(self.fStart, self.fEnd)

    def runAnimationMode(self, fInterval, fSpeedFactor):
        """
        Starts displaying images automatically (user cannot intervene).

        :param fInterval: The delay in milliseconds between adjacent frames.
        :param fSpeedFactor: Integer times the speed of the animation
        """
        self.display.runAnimation(fInterval, fSpeedFactor)
        self._loopMain()

    def runInteractiveMode(self):
        """
        Allows the user to use arrow keys to move between images.
        """
        self.display.runInteractiveMode()
        self._loopMain()

    def _loopMain(self):
        """
        Loops the GUI so it doesn't exist unless told by user.
        """
        if (sys.flags.interactive != 1) or not hasattr(QtGui, 'PYQT_VERSION'):
            QtGui.QGuiApplication.instance().exec_()
