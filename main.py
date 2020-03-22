"""

TODO


Attempt to speed up, this video acts as an introduction: https://www.youtube.com/watch?v=dPQnFXD7DxM
    This will involve making lots of helper methods that the library can use

For Radeon https://numba.pydata.org/numba-doc/dev/roc/index.html

Maybe try using OpenCL: https://documen.tician.de/pyopencl/



Back propagation:

For some reason, all images are always the same?
    Why?
Add multiple activation functions, ray loo (is that how it's spelled?), and tanh (modified sigmoid in range [-1, 1])
Implement regularization techniques:
    Using absolute value of weights for the addition to cost function, rather than the square of the weights
        Shrink weights by a constant amount, rather than based on their current weight
    Try implementing dropout, basically re-randomizing a small number of nodes
    Or, ignore randomly half the hidden nodes, then run the training through that, then do it again on
        random nodes, essentially only thinking about how some nodes effect the output at a time.
        Also remember to half the weights to account for twice as many weights being used
Apply better initialization, device each weight by the number of nodes in it's layer
Add option to not change training rate for different layers, maybe add an option to reverse it
Add learning rate scheduling, so as more training times happen, the learning rate goes down
Add test cases for new methods
Parallelize training layers, like layers should be all arrays, not a list of arrays




Image manipulation:

Try adding noise to input images, which also produce the same color output, artificially increase dataset
    Can also vertically and or horizontally flip images to increase dataset
Make a way to apply a small, say 28x28, image filter that goes along each 28x28 section of an image
    to train and process those parts of the image, maybe also have sections overlap and take averages
Make GUI and commandline program for image manipulation
Adding images to a folder should first delete the folder, then add images, to ensure extra images are not there



MNIST:

Add options to modify each data entry by shifting each image up, down, left, or right a set number of pixels.
    This allows for more data
Add options to generate random noise on the data. This allows even more data elements
Add options to rotate the images by small amounts for an increased dataset



Bugs:

Sometimes when images are loaded in MakeImages.videoToPillowImages, they are in the wrong color format,
    need to figure out places where it needs to be converted
    Fixed? Need to verify




Misc

Add a way to create a random neural network from a seed



"""


import NeuralNet.FeedForward as Net
import NeuralNet.ImageNetwork as ImgNet
from ImageManip.ColorSquares import *

import random
import time

from NeuralNet.MNIST import *


# for good results, at least temporarily
# 5000 train count, 0.05 train rate, 7 layers of 200 hidden, 16x9 in and output size, 0.1 regularization
# 0.06 train rate, 100 train count, 0.01 regularization
# 0.01 train rate, 100 train count, 70 regularization, no hidden layers, skip=30 64x36 in and out
# all above are for training2 trainFolder


inSize = (64, 36)
outSize = (64, 36)
trainCount = 1
dataSplit = 40
trainFolder = "training2"
afterPath = "images/after/"
loadNet = False
splitVideoFile = False


if splitVideoFile:
    splitVideoToInOutImages("", trainFolder, (inSize, outSize), skip=5, bars=False)
vidData = dataFromFolders(trainFolder + " (train_data)/")

if not Net.isdir("saves/after"):
    Net.mkdir("saves/after")

if loadNet:
    vidNet = Net.MatrixNetwork([])
    vidNet.load("vidNet")
else:
    vidNet = ImgNet.ImageNet(inSize, outSize, [])
    vidNet.random()


startTime = time.time()

vidNet.train(vidData, shuffle=True, split=dataSplit, times=trainCount,
             func=(lambda t, s: print("training time " + str(t) + " subset " + str(s))))

endTime = time.time() - startTime
print("Took: " + str(endTime) + " seconds")


vidNet.save("vidNet")
processFromFolder(vidNet, trainFolder + " (train_data)/grayInput/", afterPath, inSize, outSize)



"""
trainCount = 10
training = "Z:/MNIST dataset/digits/training"
testing = "Z:/MNIST dataset/digits/testing"

mnistNet = getMnistNetwork([50, 40, 30])
mnistNet.load("MIST5")
mnistNet.random()

if not trainCount == 0:
    trainData = openData(training, 10000, rand=True)
    testData = openData(testing, 100, rand=True)
    for i in range(trainCount):
        mnistNet.train(trainData, shuffle=True, split=10, times=1)
        print("(" + str(i) + ") Train correct: " + str(processData(mnistNet, trainData)) + "%")
        print("(" + str(i) + ") Test correct:  " + str(processData(mnistNet, testData)) + "%")
        print()

mnistNet.save("MIST5")

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
