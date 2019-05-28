"""
Module that contains the GUI elements of project elegancc.
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui, dockarea

import pyqtgraph.examples
# pyqtgraph.examples.run()


class ImageDisplay(QtGui.QWidget):
    """
    Responsible for rendering the images on screen using qt/pyqtgraph.
    """

    def __init__(self, fStart, fEnd, imageHandler):
        """
        :param fStart: The number of first frame of the animation.
        :param fEnd: The number of the last frame of the animation.
        :param imageHandler: An instantiated ImageHandler object that is
        responsible for reading and writing to the correct directories.
        """
        # Initialize super class and instance variables
        super(self.__class__, self).__init__()
        self.imageHandler = imageHandler
        self.fStart = fStart
        self.fEnd = fEnd
        self.frame = 1

        # Initializing window and docks
        self.initializeWindow()
        self.initializeDocks()

        # Initializing Views
        self.topImageView = self._generateView()
        self.leftImageView = self._generateView()
        self.rightImageView = self._generateView()

        # Adding ImageViews to docks
        self.topDock.addWidget(self.topImageView)
        self.leftDock.addWidget(self.leftImageView)
        self.rightDock.addWidget(self.rightImageView)

        # Setting central widget and show
        self.window.setCentralWidget(self.dockArea)
        self.window.show()

    def initializeWindow(self):
        """
        Initializes the main GUI window.
        """
        self.window = QtGui.QMainWindow()
        self.window.resize(1200, 800)
        self.window.setWindowTitle("I see Captcha!")
        self.window.setContentsMargins(0, 0, 0, 0)

    def initializeDocks(self):
        """
        Initializes the dock widgets.
        """
        # Create the docking area
        self.dockArea = pg.dockarea.DockArea()
        self.dockArea.setContentsMargins(0, 0, 0, 0)

        # Raw image dock
        self.topDock = pg.dockarea.Dock("Raw Image", size=(200, 400))
        self.topDock.setContentsMargins(0, 0, 0, 0)

        # Heat map dock
        self.leftDock = pg.dockarea.Dock("Otsu", size=(200, 400))
        self.leftDock.setContentsMargins(0, 0, 0, 0)

        # Region of interest dock
        self.rightDock = pg.dockarea.Dock("ROI (Tracking)", size=(200, 400))
        self.rightDock.setContentsMargins(0, 0, 0, 0)

        # Place the docks appropriately into the docking area
        self.dockArea.addDock(self.topDock, "top")
        self.dockArea.addDock(self.leftDock, "bottom")
        self.dockArea.addDock(self.rightDock, "right", self.leftDock)

    def initializeTimer(self, fInterval, fSpeedFactor):
        """
        Initializes the timer responsible for tracking animation timing.

        :param fInterval: The delay in milliseconds between adjacent frames.
        :param fSpeedFactor: Integer times the speed of the animation
        """
        duration = fSpeedFactor * fInterval * (self.fEnd - self.fStart)
        self.animationTimer = QtCore.QTimeLine()
        self.animationTimer.setFrameRange(self.fStart, self.fEnd)
        self.animationTimer.setUpdateInterval(fInterval)
        self.animationTimer.setDuration(duration)
        self.animationTimer.valueChanged.connect(self.updateAnimation)

    def updateAnimation(self):
        """
        Updates the animation displayed by changing the worm frame
        being displayed.
        """
        self.frame = self.animationTimer.currentFrame()
        time = self.animationTimer.currentTime()
        print("Frame: %d, Time: %.3f seconds" % (self.frame, time/1000.0))
        self.updateImages()

    def runAnimation(self, fInterval, fSpeedFactor):
        """
        Starts displaying images automatically (user cannot intervene).

        :param fInterval: The delay in milliseconds between adjacent frames.
        :param fSpeedFactor: Integer times the speed of the animation
        """
        self.initializeTimer(fInterval, fSpeedFactor)
        self.animationTimer.start()

    def runInteractiveMode(self):
        """
        Allows the user to use left/right arrow key to change between images.
        """
        def overrideArrowKeys(ev):
            if ev.key() == QtCore.Qt.Key_Right:
                self.frame += 1
            elif ev.key() == QtCore.Qt.Key_Left:
                self.frame -= 1
            self.updateImages()
        self.topImageView.keyPressEvent = overrideArrowKeys
        self.leftImageView.keyPressEvent = overrideArrowKeys
        self.rightImageView.keyPressEvent = overrideArrowKeys
        self.updateImages()

    def updateImages(self):
        """
        Updates the images displayed in all image views.
        """
        kwargs = {
            "autoRange": False,
            "autoHistogramRange": False,
            "axes": {'x': 1, 'y': 0, 'c': 2}    # Flip image
        }

        raw, _, _ = self.imageHandler.read(self.frame, "input")
        self.topImageView.setImage(raw, **kwargs)

        solved, _, _ = self.imageHandler.read(self.frame, "output/solved")
        self.rightImageView.setImage(solved, **kwargs)

        otsu, _, _ = self.imageHandler.read(self.frame, "output/otsu")
        self.leftImageView.setImage(otsu, **kwargs)

    def _generateView(self):
        """
        Generates a generic ImageView.
        """
        imageView = pg.ImageView()
        imageView.setContentsMargins(0, 0, 0, 0)
        return imageView
