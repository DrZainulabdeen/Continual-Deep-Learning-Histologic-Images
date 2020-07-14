import cv2


class ImageSegmentation:

    def __init__(self, image):
        self.image = image

    def segmentImage(self):
        ''''call model.predict() here to get the segmented classes labels here and once the model outputs 
            256*256 matrix, we can use these labels to draw a mask on original image''''

        return image, segmentation

    def addNewSegmentation(self, image, segmentation):
        ''''Add this image to database
            test if retraining should be done
            retrain if necessary''''
