"""

TODO


Back propagation

Test out code with image data



Network

Add a way of splitting a list of data into separate data groups for training batches
Make a convenient way to load two separate folders of image data into a training data set object
Make method for setting up Network based on width and height of images




Image manipulation

Make java program for making random images of the color squares to use as training data
Make GUI and commandline program for image manipulation



Bugs:

Fix issue with overflow when using somewhat large values for sigmoid



Misc

Add a way to create a random neural network from a seed



"""


import NeuralNet.FeedForward as Net
from ImageManip.MakeImages import *
from ImageManip.TrainingData import *

import random

"""
vidNet = Net.Network([10, 5, 4, 3])
vidNet.random()
vidNet.feedInputs([.1, .15, .2, .25, .3, .35, .4, .45, .5, .54])
vidNet.calculate()
# vidNet.save("test")
vidNet.load("test")
print(vidNet.getText())

"""
imgData = []
width = 32
height = 18

imgs = videoToPillowImages("", "training", size=(width, height), skip=1)

for i, im in enumerate(imgs):
    resize = scaleImage(width, height, im)

    imgData.append(imageToData(resize, width, height))
    print("Making data from: " + str(i) + " " + str(resize))


vidNet = Net.Network([len(imgData[0][0]), 20, 20, len(imgData[0][1])])
vidNet.random()


for i in range(30):
    print("Started training " + str(i))
    random.shuffle(imgData)
    vidNet.train(imgData)

for i, img in enumerate(imgData):
    vidNet.feedInputs(img[0])
    vidNet.calculate()
    outImg = dataToImage(vidNet.getOutputs(), width, height)
    outImg.save("images/trainOut" + str(i) + ".png", "PNG")
    print("saved frame " + str(i))



"""
outImg = convertGrayScale(dataToImage(imgData[0][1], width, height))
outImg.save("images/trainGray.png", "PNG")

outImg = imgs[0]
outImg.save("images/trainColor.png", "PNG")



for i, im in enumerate(imgData):
    vidNet.train(im)

    vidNet.feedInputs(imgData[0][0])
    vidNet.calculate()
    outImg = dataToImage(vidNet.getOutputs(), width, height)
    outImg.save("images/trainOut" + str(i) + ".png", "PNG")
    print("trained time " + str(i))

"""


"""
imgName = "cat.png"
width, height = 50, 50
trainCount = 40

img = Image.open("images/" + imgName)
imgData = imageToData(img, width, height)

imgNet = Net.Network([len(imgData[0]), 15, len(imgData[1])])
imgNet.random()

imgNet.feedInputs(imgData[0])
imgNet.calculate()
# imgOut = dataToImage(imgNet.getOutputs(), width, height)
# imgOut.save("images/catOut0.png", "PNG")

for i in range(trainCount):
    print("Training " + str(i))
    imgNet.train(imgData)
    # imgOut = dataToImage(imgNet.getOutputs(), width, height)
    # imgOut.save("images/catOut" + str(i + 1) + ".png", "PNG")

img = Image.open("images/cat2.png")
imgData = imageToData(img, width, height)
imgNet.feedInputs(imgData[0])
imgNet.calculate()
imgOut = dataToImage(imgNet.getOutputs(), width, height)
imgOut.save("images/cat2Out.png", "PNG")
"""

#
PRINT_EXTRA = False
TRAIN_COUNT = 100

netSize = [4, 6, 2]


data = [
    ([1, 2, 3, 4], [.5, .2]),
    ([.8, 1.6, 2, 3], [.6, .35]),
    ([.5, 1, 1.8, 2.8], [.8, .4]),
    ([.2, .8, 1.6, 2], [.85, .43]),
]
"""
data = [
    ([.1, .2, .3, .4], [.1, .4]),
    ([.2, .3, .4, .5], [.2, .5]),
    ([.3, .4, .5, .6], [.3, .6]),
    ([.4, .5, .6, .7], [.4, .7]),
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
