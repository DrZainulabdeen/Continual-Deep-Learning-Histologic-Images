from uncertaintySampling import Images


def main():
    images = Images()
    scores = images.chooseRetrainImg(images.chooseRetrainImg(pathToOldImages="/home/zain/Downloads/Thesis/unet-master/data/membrane/train/image",
                                                             pathToNewImages="/home/zain/Downloads/Thesis/unet-master/data/membrane/train/label",
                                                             technique='uncertaintySampling'))

    print(scores)


if __name__ == "__main__":
    main()
