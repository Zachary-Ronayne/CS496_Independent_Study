"""

TODO


Back propagation

Figure out how to apply the chain rule through recursive calls through backpropagate
    Also, the cost function derivative should be affected by all final output nodes, not just some?
        what does this even mean????

    Calculate the part of the derivative for each value
        that's the same across weight, bias, and activation
        this is mainly for efficiency in run time

    Make it no longer recursive
        Basically, first find the weight and bias matrix values for the output layer, which is based on the
            expected values
        Then, start at the second to last layer, and take the derivatives from that first layer
            then multiply them back, continuing the chain rule through all layers

    Make back propagation return a list of tuples,
        first part of a tuple is the 2D weight matrix
        second part of a tuple is the 1D bias matrix


Change lists to use numpy arrays rather than normal Python lists




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

"""
width = 32
height = 18
trainCount = 20

# vidData = splitVideoToFolders("", "training", (width, height), skip=30, start=0, end=300, frameRange=True)
vidData = dataFromFolders("training (train_data)/")

# vidNet = Net.makeImageNetwork(width, height, [20, 20])
# vidNet.random()
vidNet = Net.Network()
vidNet.load("vidNet")

afterPath = "images/after/"
processFromFolder(vidNet, "training (train_data)/grayInput/", afterPath, width, height)
"""

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


testTraining()