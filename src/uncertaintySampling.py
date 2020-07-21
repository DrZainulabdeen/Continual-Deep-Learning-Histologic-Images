import numpy as np
import pandas as pd
from scipy.stats import entropy
import os
import cv2
import shutil


class Images:

    def loadImages(self, paths):
        images = []
        fileNames = []
        for path in paths:  # get the names of all paths
            for filename in os.listdir(path):  # for every path
                if filename.endswith(".png"):
                    img = cv2.imread(os.path.join(path, filename))
                    if img is not None:  # If image is not null append it to list
                        images.append(img)
                        fileNames.append(filename)
        return fileNames, images

    def chooseRetrainImg(self, pathToOldImages, pathToNewImages, technique):
        self.pathToOldImages = pathToOldImages
        self.pathToNewImages = pathToNewImages
        self.technique = technique

        score = []
        paths = [pathToOldImages, pathToNewImages]

        fileNames, images = self.loadImages(paths)

        if(technique == 'uncertaintySampling'):

            for image in images:  # calculate entropies for all images
                _, counts = np.unique(image, return_counts=True)
                score.append(entropy(counts))

        chosenImages = sorted(zip(fileNames, score), key=lambda x: x[1])

        return chosenImages  # TODO----->Still need to chose how many images

    def updateDatabase(self, chosenImages):
        source = 'Path of new images'
        destination = 'Path of old images'

        for image in chosenImages:
            shutil.move(os.path.join(source, image[0]), destination)

    def pred_probability(counts):
        prob = []
        for sample in counts:
            n_classes = len(sample)
            prob.append(sample/n_classes)
        return prob
