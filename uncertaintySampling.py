import numpy as np
import pandas as pd
from scipy.stats import entropy
import os
import cv2


class Images:

    def __init__(self):
        pass

    def loadImages(self, paths):
        self.paths = paths
        images = []

        for path in paths:  # get the names of all paths
            for filename in os.listdir(path):  # for every path
                if filename.endswith(".png"):
                    img = cv2.imread(os.path.join(path, filename))
                    if img is not None:  # If image is not null append it to list
                        images.append(img)
        return images

    def chooseRetrainImg(self, pathToOldImages, pathToNewImages, technique):
        self.pathToOldImages = pathToOldImages
        self.pathToNewImages = pathToNewImages
        self.technique = technique

        score = []
        paths = [pathToOldImages, pathToNewImages]

        images = self.loadImages(paths)

        if(technique == 'uncertaintySampling'):

            for image in images:  # calculate entropies for all images
                classes, counts = np.unique(image, return_counts=True)
                score.append(entropy(counts))

        score.sort()

        return score
