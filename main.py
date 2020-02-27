"""

TODO


Back propagation

Test out code and make sure backpropagation works all as intended
    for some reason, all images are always the same?
Add thing to automatically shuffle test data
Change lists to use numpy arrays rather than normal Python lists
Make separate backpropagation method that uses matrix multiplication
    will also need to set up a conversion thing to transform a Network object into just two objects for weights and bias




Network

Make a class extending Network, specifically with extra functionality for processing images and saving things
    like height and width




Image manipulation

Make program for making random images of the color squares to use as training data
Make GUI and commandline program for image manipulation
Add option to TrainingData.scaleImage for resizing based on adding black bars, or stretching



Bugs:

Sometimes when images are loaded in MakeImages.videoToPillowImages, they are in the wrong color format,
    need to figure out places where it needs to be converted



Misc

Add a way to create a random neural network from a seed



"""


import NeuralNet.FeedForward as Net
from ImageManip.TrainingData import *
from ImageManip.ColorSquares import *
import random


# saveSquareTrainingData("colorSquareTest", "", 100, size=5, width=5, height=5)


"""
matrixNet = Net.MatrixNetwork([2, 3, 4, 5])
matrixNet.random()

print(matrixNet.getText())

matrixNet.save("test")
matrixNet.load("test")
"""

width = 25
height = 25
trainCount = 5
dataSplit = 1
trainFolder = "colorSquareTest"
afterPath = "images/after/"
loadNet = True


# splitVideoToInOutImages("", trainFolder, (width, height), skip=2)
vidData = dataFromFolders(trainFolder + " (train_data)/")

if loadNet:
    vidNet = Net.Network()
    vidNet.load("vidNet")
else:
    vidNet = Net.makeImageNetwork(width, height, [20])
    vidNet.random()

for i in range(trainCount):
    random.shuffle(vidData)
    split = dataSubSet(vidData, dataSplit)
    for j, s in enumerate(split):
        print("training time " + str(i) + " subset " + str(j))
        vidNet.train(s)


vidNet.save("vidNet")
processFromFolder(vidNet, trainFolder + " (train_data)/grayInput/", afterPath, width, height)


#
PRINT_EXTRA = False
TRAIN_COUNT = 200

netSize = [4, 6, 2]

data = [
    ([1, 2, 3, 4], [.5, .2]),
    ([.8, 1.6, 2, 3], [.6, .35]),
    ([.5, 1, 1.8, 2.8], [.8, .4]),
    ([.2, .8, 1.6, 2], [.85, .43]),
]
"""
data = [
    ([1, 1, 1, 1], [.5, .5]),
    ([.5, .5, .5, .5], [.25, .25]),
    ([.51, .51, .51, .51], [.26, .26]),
    ([.1, .1, .1, .1], [.05, .05]),
]
"""


def printData(net):
    for i in range(len(data)):
        if PRINT_EXTRA:
            print("expected:")
            print(data[i][1])

        if PRINT_EXTRA:
            print("raw:")
        net.feedInputs(data[i][0])
        net.calculate()
        outs = net.getOutputs()
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
    net = Net.Network(netSize)
    net.random()

    print("\nBefore:\n")
    printData(net)

    for ii in range(TRAIN_COUNT):
        net.train(data)

    print("\nAfter:\n")
    printData(net)


# testTraining()
