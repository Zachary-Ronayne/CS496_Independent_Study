# A file for handling loading in and processing the MNIST dataset with a neural network

from os.path import isfile, join
from os import listdir

from PIL import Image

from ImageManip.MakeImages import scaleImage
from ImageManip.TrainingData import grayImageToData
from NeuralNet.FeedForward import *

import Settings

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
        if Settings.IMG_PRINT_STATUS:
            print("opening MNIST files " + n)

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


def getMnistNetwork(inner):
    """
    Get a network object for processing MNIST data
    :param inner: The inner nodes. [25, 20], would give an extra layer of 25, then 20 nodes between the input and
        output layers
    :return: The Network
    """
    # add the input size to the beginning, add the output size to the end
    inner.insert(0, MNIST_SIZE * MNIST_SIZE)
    inner.append(10)
    # return the correct MatrixNetwork
    return MatrixNetwork(inner)


def getOutputData(num):
    """
    Get the output data list for the given number
    :param num: The number, must be in range [0-9]
    :return: The list of output data. A list of 10 numbers, all are 0, except the one matching the given number
    """
    # make a list of 10 zeros
    data = [0] * 10
    # set the appropriate element to 1
    data[num] = 1
    # return the data list
    return data


def chooseOutput(outputs):
    """
    Get the chosen output for the MNIST Network output
    :param outputs: The outputs of the Network, should be 10
    :return: the highest entry in the output
    """
    # initialize the highest index to 0
    high = 0
    # iterate through each output value
    for i, n in enumerate(outputs):
        # if the current iteration is higher than the highest saved, save the new highest
        if n > outputs[high]:
            high = i

    # return the new highest index, which is the numerical choice the Network makes
    return high


def processData(net, data):
    """
    Use the given Network to process the given data, and get the percentage of correctly classified digits
    :param net: The Network to use
    :param data: The data to use, should be a list of data, each entry is a tuple of the form (input, expect output)
    :return: The percentage correct
    """

    # initialize the number of correct examples to 0
    correct = 0
    # go through each piece of data
    for d in data:
        # get the outputs
        outs = net.calculateInputs(d[0])
        # if the outputs are equal to the expected outputs, then increment the counter
        if chooseOutput(outs) == chooseOutput(d[1]):
            correct += 1

    # return the percentage correct
    return 100.0 * correct / len(data)


def processHandWritten(net, path):
    """
    Use the given Network to process the image at the given path
    :param net: The Network to use
    :param path: The path to the image, this should be the entire file, extension and all
    :return: The Network's guess as to what the result is
    """
    # load the image
    img = Image.open(path)
    # resize the image
    img = scaleImage(28, 28, img)
    # get the data from the image
    data = grayImageToData(img)
    # determine the outputs of the image with the net
    outs = net.calculateInputs(data)
    # return the chosen number
    return chooseOutput(outs)
