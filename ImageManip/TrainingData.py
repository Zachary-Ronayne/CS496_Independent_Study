from ImageManip.MakeImages import *

import shutil
from os.path import isdir


def folderToInOutImages(inSize, outSize, source, folder, bars=True):
    """
    Take all the images in a folder and generate training data images for all images in the given source.
    The data is split into two folders, one is the gray scale input, in a folder called grayInput.
    The other is the colored expected output in a folder called colorOutput.
    All images will be the same aspect ratio as the originals, adding black bars to the sides or top
        if extra space needs to be filled.
    All images will be the same number of pixels in width and height.
    :param inSize: A tuple of (width, height) for the number of pixels in the width and height of the input images
            must be a positive integer > 0
    :param outSize: A tuple of (width, height) for the number of pixels in the width and height of the output images
            must be a positive integer > 0
    :param source: The path to the folder containing all images. The folder must only contain images.
        The folder is relative to the images directory
        or
        A list of Pillow images to convert
    :param folder: The folder to save the images to, relative to images.
        Will automatically add on " (train_data)" to the folder to seperate it from other folders
    :param bars: True if images of differing aspect ratios should apply black bars to fill the space,
        False to stretch the image. Default True
    """
    # direct the folder to the images folder
    folder += " (train_data)"
    folder = "images/" + folder + "/"

    # create the output folder, deleting it first if it already exists
    if isdir(folder):
        shutil.rmtree(folder)
    mkdir(folder)

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
        i = scaleImage(outSize[0], outSize[1], i)
        # save the color image
        i.save(colorPath + "color" + num + ".png", "PNG")
        if Settings.IMG_PRINT_STATUS:
            print("saved color image: " + str(cnt) + " " + str(i))

        # create the gray image
        i = convertGrayScale(i)
        # resize the image
        i = scaleImage(inSize[0], inSize[1], i, bars=bars)
        # save the gray image
        i.save(grayPath + "gray" + num + ".png", "PNG")
        if Settings.IMG_PRINT_STATUS:
            print("saved gray image: " + str(cnt) + " " + str(i))

        cnt += 1


def dataFromFolders(path):
    """
    Get a list of training data from a folder with the format created by trainingDataFromFolder
    the folder should contain a folder called grayInput for the input images,
       and a folder called colorOutput for the expected output.
    Both the folders should contain the exact same number of images, all of which are formatted
       correctly for training data.
    The images in the folders should only contain the training data and no other folders or files
    :param path: The path to the two folders, relative to images
    :return: the data
    """
    # ensure that the saves folder exists
    createImages()

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


def splitVideoToInOutImages(path, name, sizes=(None, None), skip=1, start=0, end=1, frameRange=False, bars=True):
    """
    Take the video file at the given path and create a folder of images for input data,
        and a folder of images for output data.
    :param path: The folder path containing the video file, relative to images
    :param name: The file name of the video, including file extension
    :param sizes: A tuple of two tuples, each in the form of (width, height) for the number of pixels in images.
            The 0 index is the input image size, the 1 index is the output image size
    :param skip: Skip every this many frames, default 1, meaning skip no frames
    :param start: The percentage in range [0, 1] of the starting point in the video to produce images, must be < end
    :param end: The percentage in range [0, 1] of th end point in the video tp produce images, must be > start
            can also use integers for start and end along with the flag frameRange set to true to use a range of frames for
            start and end.
    :param frameRange: True to make start and end act as frame ranges, False to make them act as percentage ranges
    :param bars: True if images of differing aspect ratios should apply black bars to fill the space,
        False to stretch the image. Default True
    :return: The training data as a tuple
    """

    # remove file extention
    fileName = name[:name.index(".")]

    # ensure that the images folder exists
    createImages()

    # determine the new path name for where each folder will be saved
    trainingDataPath = path + fileName + " (train_data)/"

    if isdir("images/" + trainingDataPath):
        shutil.rmtree("images/" + trainingDataPath)

    # create a folder with each frame of the video, and store the path
    splitPath = splitVideo(path, name, sizes[1], skip, start, end, frameRange, bars=bars)

    # make a directory for the new folder
    if not isdir("images/" + trainingDataPath):
        mkdir("images/" + trainingDataPath)

    # get training data from the split frames and convert them to images
    folderToInOutImages(sizes[0], sizes[1], splitPath, path + fileName, bars=bars)

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


def processFromFolder(net, path, outPath, inSize, outSize):
    """
    Using the given Network, calculate and save images for each image in the given path
    :param net: The Network to process the images
    :param path: A folder, relative to images, of the gray scale images to use for input
    :param outPath: A folder, relative to images, to where each image should be saved
    :param inSize: A tuple of (width, height) of the size of images that the Network processes
    :param outSize: A tuple of (width, height) of the size of images that the Network outputs
    """

    # ensure that the saves folder exists
    createImages()

    # load in all gray scale images
    path = "images/" + path
    data = [f for f in listdir(path)]
    for i in range(len(data)):
        # load image
        data[i] = Image.open(path + data[i])

        # resize appropriately
        data[i] = scaleImage(inSize[0], inSize[1], data[i])

        # convert to data
        data[i] = grayImageToData(data[i])

    # determine and save all images
    for i, d in enumerate(data):
        # get the output data of the Network
        img = dataToImage(net.calculateInputs(d), outSize[0], outSize[1])
        # save the output data as an image
        img.save(outPath + "output " + str(i) + ".png", "PNG")


def createImages():
    """
    Create the directory for the images folder if one doesn't exist
    """
    if not isdir("images"):
        mkdir("images")


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
    # get the pixels
    pixels = img.load()
    width, height = img.size

    # return the input data, getting every pixel in each column, then each row
    if img.mode == "L":
        return [pixels[x, y] / 256.0 for x in range(width) for y in range(height)]
    else:
        return [pixels[x, y][0] / 256.0 for x in range(width) for y in range(height)]


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


def dataSubSet(data, size):
    """
    Take a single list of data and divide it into lists of equal size
    :param data: The list of items to separate
    :param size: The number of elements that should be in each sub list, any elements that don't fit evenly will be placed in
                one list smaller than the rest of the lists
    :return: The subsets of data
    """
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
