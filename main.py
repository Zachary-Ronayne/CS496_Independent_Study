"""

TODO


Attempt to speed up, this video acts as an introduction: https://www.youtube.com/watch?v=dPQnFXD7DxM
    This will involve making lots of helper methods that the library can use



Back propagation:

For some reason, all images are always the same?
Add test cases for new methods
Allow the MatrixNetwork to use the max and min values for weight and bias
Try adding multiple options for cost functions



Image manipulation:

Make GUI and commandline program for image manipulation




MNIST:

Add options to modify each data entry by shifting each image up, down, left, or right a set number of pixels.
    This allows for more data
Add options to generate random noise on the data. This allows even more data elements




Bugs:

Sometimes when images are loaded in MakeImages.videoToPillowImages, they are in the wrong color format,
    need to figure out places where it needs to be converted

Overflow in FeedForward.sigmoid(x)

When not using the exact same aspect ratio, input and output images of different sizes don't work



Misc

Add a way to create a random neural network from a seed



"""


import NeuralNet.FeedForward as Net
import NeuralNet.ImageNetwork as ImgNet
from ImageManip.ColorSquares import *
import random

from NeuralNet.MNIST import *


inSize = (32, 32)
outSize = (32, 18)
trainCount = 100
dataSplit = 40
trainFolder = "training"
afterPath = "images/after/"
loadNet = True
splitVideoFile = False

if splitVideoFile:
    splitVideoToInOutImages("", trainFolder, (inSize, outSize), skip=10, bars=False)
vidData = dataFromFolders(trainFolder + " (train_data)/")

if loadNet:
    vidNet = Net.MatrixNetwork([])
    vidNet.load("vidNet")
else:
    vidNet = ImgNet.ImageNet(inSize, outSize, [100, 100])
    vidNet.random()

vidNet.train(vidData, shuffle=True, split=dataSplit, times=trainCount,
             func=(lambda t, s: print("training time " + str(t) + " subset " + str(s))))

vidNet.save("vidNet")
processFromFolder(vidNet, trainFolder + " (train_data)/grayInput/", afterPath, inSize, outSize)


"""
trainCount = 0
training = "Z:/MNIST dataset/digits/training"
testing = "Z:/MNIST dataset/digits/testing"

mnistNet = getMnistNetwork([140, 100, 80, 60,  50])
mnistNet.load("MIST6")

if not trainCount == 0:
    trainData = openData(training, 10000, rand=True)
    testData = openData(testing, 100, rand=True)
    for i in range(trainCount):
        mnistNet.train(trainData, shuffle=True, split=10, times=1)
        print("(" + str(i) + ") Train correct: " + str(processData(mnistNet, trainData)) + "%")
        print("(" + str(i) + ") Test correct:  " + str(processData(mnistNet, testData)) + "%")
        print()

mnistNet.save("MIST6")

imgFile = "Z:/MNIST dataset/num.png"
print(processHandWritten(mnistNet, imgFile))
"""

"""
PRINT_EXTRA = False
TRAIN_COUNT = 2000

netSize = [4, 6, 2]

data = [
    ([1, 2, 3, 4], [.5, .2]),
    ([.8, 1.6, 2, 3], [.6, .35]),
    ([.5, 1, 1.8, 2.8], [.8, .4]),
    ([.2, .8, 1.6, 2], [.85, .43]),
]
data = [
    ([1, 1, 1, 1], [.5, .5]),
    ([.5, .5, .5, .5], [.25, .25]),
    ([.51, .51, .51, .51], [.26, .26]),
    ([.1, .1, .1, .1], [.05, .05]),
]


def printData(net):
    for i in range(len(data)):
        if PRINT_EXTRA:
            print("expected:")
            print(data[i][1])

        if PRINT_EXTRA:
            print("raw:")
        outs = net.calculateInputs(data[i][0])
        if PRINT_EXTRA:
            print(outs)

            print("differences:")
        diffs = []
        errors = []
        for j in range(len(data[i][1])):
            diffs.append(round(abs(outs[j] - data[i][1][j]), 4))
            errors.append(round(100 * diffs[j] / outs[j], 4))
        if PRINT_EXTRA:
            print(diffs)

        print("% error")
        s = ""
        for e in errors:
            s += str(e) + "% "
        print(s)


def printGradient(gradient):
    print("\nGradient:\n")
    tabs = 0
    for c in str(gradient):
        if c == "]":
            tabs -= 1
            print()
            for i in range(tabs):
                print("\t", end="")
        elif c == "[":
            tabs += 1
            print()
            for i in range(tabs):
                print("\t", end="")
        print(c, end="")

    print()


def testTraining():
    net = Net.MatrixNetwork(netSize)
    net.random()

    print("\nBefore:\n")
    printData(net)

    for ii in range(TRAIN_COUNT):
        net.train(data)

    print("\nAfter:\n")
    printData(net)


# testTraining()

"""
