from ImageManip.MakeImages import *

import shutil


def folderToInOutImages(width, height, source, folder):
    """
    Take all the images in a folder and generate training data images for all images in the given source.
    The data is split into two folders, one is the gray scale input, in a folder called grayInput.
    The other is the colored expected output in a folder called colorOutput.
    All images will be the same aspect ratio as the originals, adding black bars to the sides or top
        if extra space needs to be filled.
    All images will be the same number of pixels in width and height.
    :param width: The number of pixels in the width of the output images, must be a positive integer > 0
    :param height: The number of pixels in the height of the output images, must be a positive integer > 0
    :param source: The path to the folder containing all images. The folder must only contain images.
        The folder is relative to the images directory
        or
        a list of Pillow images to convert
    :param folder: The folder to save the images to, relative to images
    :return:
    """
    # direct the folder to the images folder
    folder = "images/" + folder + "/"

    # create the paths for the folders
    colorPath = folder + "colorOutput/"
    grayPath = folder + "grayInput/"

    # if the folders do not already exist, create them
    if not isdir(colorPath):
        mkdir(colorPath)

    if not isdir(grayPath):
        mkdir(grayPath)

    # get all the files in the given folder
    if isinstance(source, list):
        images = source
    else:
        images = []
        files = [file for file in listdir(source) if isfile(join(source, file))]

        # for each image, create the color image, then save it and convert it to gray and save it as well
        for f in files:
            # create the color image
            img = Image.open(source + f)
            images.append(img)

    # process all images into the training data and save them
    cnt = 0
    for i in images:
        num = str(cnt)
        while len(num) < Settings.IMG_NUM_DIGITS:
            num = "0" + num

        # create the color image
        i = scaleImage(width, height, i)
        # save the color image
        i.save(colorPath + "color" + num + ".png", "PNG")
        if Settings.IMG_PRINT_STATUS:
            print("saved color image: " + str(cnt) + " " + str(i))

        # create the gray image
        i = convertGrayScale(i)
        # save the gray image
        i.save(grayPath + "gray" + num + ".png", "PNG")
        if Settings.IMG_PRINT_STATUS:
            print("saved gray image: " + str(cnt) + " " + str(i))

        cnt += 1


# get a list of training data from a folder with the format created by trainingDataFromFolder
# the folder should contain a folder called grayInput for the input images,
#   and a folder called colorOutput for the expected output
# both the folders should contain the exact same number of images, all of which are formatted
#   correctly for training data
# the images in the folders should only contain the training data and no other folders or files
# path: the path to the two folders, relative to images
def dataFromFolders(path):
    # determine folder paths
    path = "images/" + path
    grayPath = path + "grayInput/"
    colorPath = path + "colorOutput/"

    # load in gray images
    grayImg = [f for f in listdir(grayPath)]
    for i in range(len(grayImg)):
        grayImg[i] = Image.open(grayPath + grayImg[i])

    # load in color images
    colorImg = [f for f in listdir(colorPath)]
    for i in range(len(colorImg)):
        colorImg[i] = Image.open(colorPath + colorImg[i])

    data = []

    # add all the data tuples into the data list
    for i, img in enumerate(grayImg):
        data.append((grayImageToData(img), colorImageToData(colorImg[i])))

    return data


# take the video file at the given path and create a folder of images for input data,
#   and a folder of images for output data
# also returns the training data as a tuple
# path: the folder path containing the video file, relative to images
# name: the file name of the video, excluding file extension, must be .mov
# see MakeImages.videoToImages for extra parameters description
def splitVideoToInOutImages(path, name, size=None, skip=1, start=0, end=1, frameRange=False):
    # create a folder with each frame of the video, and store the path
    splitPath = splitVideo(path, name, size, skip, start, end, frameRange)
    # determine the new path name for where each folder will be saved
    trainingDataPath = path + name + " (train_data)/"

    # make a directory for the new folder
    if not isdir("images/" + trainingDataPath):
        mkdir("images/" + trainingDataPath)

    # get training data from the split frames and convert them to images
    folderToInOutImages(size[0], size[1], splitPath, trainingDataPath)

    # delete the folder from the initial video file split
    shutil.rmtree(splitPath)

    # generate the data from the newly created images
    return dataFromFolders(trainingDataPath)


# train the given Network on the images in the given path
# net: the Network that will be trained
# path: the path, relative to images, containing the two folders for input and output data, grayInput and colorInput
# times: number of times to train on the data
def trainOnFolder(net, path, times=1):
    # get the data
    data = dataFromFolders(path)

    # train on the given data the specified number of times
    for i in range(times):
        net.train(data)


# using the given Network, calculate and save images for each image in the given path
# net: the Network to process the images
# path: a folder, relative to images, of the gray scale images to use for input
# outPath: a folder, relative to images, to where each image should be saved
# width: the width of the images that the Network processes
# height: the height of the images that the Network processes
def processFromFolder(net, path, outPath, width, height):
    # load in all gray scale images
    path = "images/" + path
    data = [f for f in listdir(path)]
    for i in range(len(data)):
        # load image
        data[i] = Image.open(path + data[i])

        # resize appropriately
        data[i] = scaleImage(width, height, data[i])

        # convert to data
        data[i] = grayImageToData(data[i])

    # determine and save all images
    for i, d in enumerate(data):
        # determine the output of the Network with the current image
        net.feedInputs(d)
        net.calculate()
        # get the output data of the Network
        img = dataToImage(net.getOutputs(), width, height)
        # save the output data as an image
        img.save(outPath + "output " + str(i) + ".png", "PNG")


# take a PIL image and return a tuple of input and output data for an image Neural Network
# the first element is the input data, the second element is the expect output
# img: the PIL image to process
# width: the desired with of the resized version of the image
# height: the desired height of the resized version of the image
def imageToData(img, width, height):
    # create the reszied color image
    color = scaleImage(width, height, img)
    # add all the color values to the expected output data
    exData = colorImageToData(color)

    # create the gray scale version of the color image
    gray = convertGrayScale(color)
    # add all the color values to the input data
    inData = grayImageToData(gray)
    return inData, exData


# take a color PIL image and convert it to expected output training data
def colorImageToData(img):
    exData = []
    # get the pixels
    pixels = img.load()
    width, height = img.size
    # go through each pixel and add the data for each color channel
    for x in range(width):
        for y in range(height):
            for c in range(3):
                exData.append(pixels[x, y][c] / 256.0)

    return exData


# take a gray scale PIL image and convert it to input training data
# this method assumes all 3 color channels have the same value
def grayImageToData(img):
    inData = []
    # get the pixels
    pixels = img.load()
    width, height = img.size
    # go through each pixel and add the data for the gray
    for x in range(width):
        for y in range(height):
            # assume all channels are the same value, so just take the red value
            inData.append(pixels[x, y][0] / 256.0)

    # return the input data
    return inData


# get a PIL image from the given list of data
# data: a 1 dimensional list of all the RGB values, first element is red, then green, then blue, then next pixel
#   the data should be in range [0, 1], which will be converted to [0, 256]
# width: the number of pixels in a row
# height: the number of pixels in a column
def dataToImage(data, width, height):
    # make a new image of the correct size
    img = Image.new("RGB", (width, height))

    cnt = 0
    # go through each pixel and set each color channel based on the data in the list
    for x in range(width):
        for y in range(height):
            img.putpixel((x, y),
                         (int(round(data[cnt] * 256)),
                          int(round(data[cnt + 1] * 256)),
                          int(round(data[cnt + 2] * 256)),
                          256))

            cnt += 3

    return img


# take a single list of data and divide it into lists of equal size
# data: the list of items to separate
# size: the number of elements that should be in each sub list, any elements that don't fit evenly will be placed in
#   one list smaller than the rest of the lists
def dataSubSet(data, size):
    retData = []

    # go through the list of data in increments of the size of list divisions
    for i in range(0, len(data), size):
        end = i + size
        # if the new index for the next section is outside the list range, then set the end index to the end of the list
        if end > len(data) - 1:
            end = len(data)
        # add the next section of the original list
        retData.append(data[i:end])

    return retData
