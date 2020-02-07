"""

TODO


Back propagation

Test out code with image data



Image manipulation

Add settings for selecting a range of a video file, time codes and percentages
Add settings for selecting every certain number of frames
Make GUI and commandline program for image manipulation



Misc

Add a way to create a random neural network from a seed



"""


import NeuralNet.FeedForward as Net


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


def printData():
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


net = Net.Network(netSize)
net.random()

print("\nBefore:\n")
printData()

for ii in range(TRAIN_COUNT):
    net.train(data)

print("\nAfter:\n")
printData()
