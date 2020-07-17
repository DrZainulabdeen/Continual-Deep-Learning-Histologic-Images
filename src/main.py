from uncertaintySampling import Images


def main():
    images = Images()
    chosenImages = images.chooseRetrainImg(pathToOldImages="/home/zain/Downloads/Thesis/unet-master/data/membrane/train/image",
                                                             pathToNewImages="/home/zain/Downloads/Thesis/unet-master/data/membrane/train/label",
                                                             technique='uncertaintySampling')

    print(chosenImages)

'''    if(checkTrain()):
        if(checkNewImages()):
            chosenImages = images.chooseRetrainImg()

            #Define a threshold here for how many images we want to retrain or maybe only return chosenImages
            #if it has more than threshold images with good information

            if(chosenImages<= #Somenumber):
                selectBatchSize()
                retrain()
                updateDatabase(chosenImages)
'''



if __name__ == "__main__":
    main()
