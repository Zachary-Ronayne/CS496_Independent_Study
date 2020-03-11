# A file for handling loading in and processing the MNIST dataset with a neural network

from os.path import isfile, join
from os import listdir

from PIL import Image

from ImageManip.TrainingData import grayImageToData
from NeuralNet.FeedForward import *

# The size, in pixels of an MNIST dataset image
MNIST_SIZE = 28


def openData(path, limit=None, rand=False):
    """
    Get a list of data entries of the form (input, output) from the stored locations of the MNIST dataset
    :param path: The folder location containing folders of images. Each folder is a single number character from 0-9.
        The images in these folders should be the correct digit of the corresponding folder.
        All images must be 28x28 pixels
    :param limit: The maximum number of images to load from each folder, None to load all, default: None
    :param rand: True to randomize which images are selected from the folder and their order, False otherwise,
        default: False
    :return: The training data
    """

    # initialize data list
    data = []
    # set up a list of all the folders to search through
    nums = [str(i) for i in range(0, 10)]

    # iterate through each folder
    for i, n in enumerate(nums):
        # get the path of the folder
        numPath = path + "/" + n
        # get all the images in the folder
        files = ["/".join([numPath, file]) for file in listdir(numPath)]

        # shuffle the files if applicable
        if rand:
            random.shuffle(files)

        # iterate through each image, up to the limit number of images
        for cnt, f in enumerate(files):
            # end the loop early if the limit for images is reached
            if limit is not None and cnt >= limit:
                break
            # load in the PIL image
            img = Image.open(f)

            # convert the image to input data
            dat = grayImageToData(img)

            # add the data entry to the data list
            data.append((dat, getOutputData(i)))

    return data


def getOutputData(num):
    """
    Get the output data list for the given number
    :param num: The number, must be in range [0-9]
    :return: The list of output data. A list of 10 numbers, all are 0, except the one matching the given number
    """
    data = [0] * 10
    data[num] = 1
    return data


def getMnistNetwork(inner):
    """
    Get a network object for processing MNIST data
    :param inner: The inner nodes. [25, 20], would give an extra layer of 25, then 20 nodes between the input and
        output layers
    :return: The Network
    """

    inner.insert(0, MNIST_SIZE * MNIST_SIZE)
    inner.append(10)
    return MatrixNetwork(inner)


def chooseOutput(outputs):
    """
    Get the chosen output for the MNIST Network output
    :param outputs: The outputs of the Network, should be 10
    :return: the highest entry in the output
    """
    high = 0
    for i, n in enumerate(outputs):
        if n > outputs[high]:
            high = i

    return high
