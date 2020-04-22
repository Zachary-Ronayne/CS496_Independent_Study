"""

TODO

Back propagation:

Apply better initialization, device each weight by the number of nodes in it's layer
Investigate convolutional layers
    Add another matrix in each layer, and that is like a filter, it goes over each section of the image
    Basically take the dot product at each spot, and place that in a new array for the image
    The output sizes wont necessarily be the same?
    The filters have esentially random numbers, basically the same kind of thing as a weight
    High numbers mean the filter finds something important, low numbers mean it's something not important
    The filter replaces each node



Image manipulation:
Change image splitting to stretch images, not fill them with black space, give option to remove them also
Allow a folder of images to be split into images for the split images thing, like the small filter
Make video files load in frames more efficiently, only load in necessary frames, not every frame and then skipping some
Try adding noise to input images, which also produce the same color output, artificially increase dataset
    Can also vertically and or horizontally flip images to increase dataset
Add a system to use overlapping parts in the image splitter




MNIST:

Add options to modify each data entry by shifting each image up, down, left, or right a set number of pixels.
    This allows for more data
Add options to generate random noise on the data. This allows even more data elements
Add options to rotate the images by small amounts for an increased dataset



Bugs:

Sometimes when images are loaded in MakeImages.videoToPillowImages, they are in the wrong color format,
    need to figure out places where it needs to be converted
    Fixed? Need to verify

Normalize the formats for saving and loading images in folders, it's currently very hacky



Misc

Add a way to create a random neural network from a seed



Parallelization (need NVIDIA GPU):

Attempt to speed up, this video acts as an introduction: https://www.youtube.com/watch?v=dPQnFXD7DxM
    This will involve making lots of helper methods that the library can use

For Radeon? https://numba.pydata.org/numba-doc/dev/roc/index.html

Maybe try using OpenCL: https://documen.tician.de/pyopencl/



"""


import NeuralNet.FeedForward as Net
import NeuralNet.ImageNetwork as ImgNet
from ImageManip.ColorSquares import *

import random
import time

from NeuralNet.MNIST import *
from NeuralNet.Convolutional import *

from ImageManip.ImageSpliter import *


trainCount = 1
training = "Z:/MNIST dataset/digits/training"
testing = "Z:/MNIST dataset/digits/testing"

conNet = Convolution([(4, 2), (4, 2), (4, 2)], 10)
conNet.random()

if not trainCount == 0:
    trainData = squareData(openData(training, 5, rand=True))
    testData = squareData(openData(testing, 5, rand=True))
    for i in range(trainCount):
        conNet.train(trainData, shuffle=True, split=10, times=1)
        print("(" + str(i) + ") Train correct: " + str(processData(conNet, trainData)) + "%")
        print("(" + str(i) + ") Test correct:  " + str(processData(conNet, testData)) + "%")
        print()


"""

# for good results, at least temporarily
# 5000 train count, 0.05 train rate, 7 layers of 200 hidden, 16x9 in and output size, 0.1 regularization
# 0.06 train rate, 100 train count, 0.01 regularization
# 0.01 train rate, 100 train count, 70 regularization, no hidden layers, skip=30 64x36 in and out
# all above are for training2 trainFolder

mnist = False

if not mnist:

    imgSize = 20
    vidData = []
    # for i in range(7):
    #    splitImage("cats/cat" + str(i) + ".png", "trainingCat" + str(i), imgSize, imgSize, resize=(320, 180))
    for i in range(7):
        vidData.extend(dataFromFolders("trainingCat" + str(i) + " (train_data)/"))

    # inSize = (64, 36)
    # outSize = (64, 36)
    inSize = (imgSize, imgSize)
    outSize = (imgSize, imgSize)
    trainCount = 100
    dataSplit = 10
    trainFolder = "trainingCat"
    trainName = trainFolder + ".mov"
    afterPath = "images/after/"
    loadNet = False
    splitVideoFile = False
    process = True

    if splitVideoFile:
        splitVideoToInOutImages("", trainName, (inSize, outSize), skip=15, bars=False)
    # vidData = dataFromFolders(trainFolder + " (train_data)/")

    if not Net.isdir("images/after"):
        Net.mkdir("images/after")

    if loadNet:
        vidNet = Net.MatrixNetwork([])
        vidNet.load("vidNet")
    else:
        vidNet = ImgNet.ImageNet(inSize, outSize, [5000])
        vidNet.random()

    if trainCount > 0:
        startTime = time.time()
        vidNet.train(vidData, shuffle=True, split=dataSplit, times=trainCount, learnSchedule=-.5,
                     func=(lambda t, s: print("training time " + str(t) + " subset " + str(s))))
        endTime = time.time() - startTime
        print("Took: " + str(endTime) + " seconds")

    vidNet.save("vidNet")
    if process:
        processFromFolder(vidNet, trainFolder + " (train_data)/grayInput/", afterPath, inSize, outSize)

    cat = Image.open("images/cat.png")
    cat = convertGrayScale(cat)
    imgNet = ImgNet.convertFromMatrix(vidNet, inSize, outSize)
    finalImg = applyNetwork(cat, imgNet, resize=None)
    finalImg.save("images/catNew.png")

else:
    trainCount = 100
    training = "Z:/MNIST dataset/digits/training"
    testing = "Z:/MNIST dataset/digits/testing"

    mnistNet = getMnistNetwork([50, 40, 30])
    mnistNet.load("MIST5")

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
