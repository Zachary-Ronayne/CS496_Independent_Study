"""

TODO


Back propagation

Make cohesive pseudo code
Add test cases for backprogpgate and related methods
Allow for multiple training examples to be used all at once



Image manipulation

Add settings for selecting a range of a video file, time codes and percentages
Add settings for selecting every certain number of frames
Make GUI and commandline program for image manipulation



Misc

Add a way to create a random neural network from a seed



"""


import NeuralNet.FeedForward as Net

netSize = [4, 6, 2]
net = Net.Network(netSize)
net.random()

data = [
    ([1, 2, 3, 4], [.5, .2]),
    ([4, 3, 2, 1], [.1, .7]),
]

net.feedInputs(data[0][0])
net.calculate()

print("\nBefore:\n")
print(net.getText())


"""
gradient = net.backpropagate(len(netSize) - 1, [.5, .2], [])

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
"""

for i in range(100):
    net.train(data)

print("\nAfter:\n")
net.feedInputs(data[0][0])
net.calculate()
print(net.getOutputs())
net.feedInputs(data[1][0])
net.calculate()
print(net.getOutputs())
