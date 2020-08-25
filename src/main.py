from uncertaintySampling import Images


def main():
    images = Images()
    #oldPath = '/home/GeorgHille/zain/data/Images'
    #newPath = '/home/GerogHille/zain/data/newImages'
    oldPath = '/home/zain/seafile-client/seafile/Zain/Images'
    newPath = '/home/zain/seafile-client/seafile/Zain/newImages'
    chosenImages = images.chooseRetrainImg(
        oldPath, newPath, technique='uncertaintySampling')

    print(chosenImages[0])


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
