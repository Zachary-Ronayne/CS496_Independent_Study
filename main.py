"""

TODO


Back propagation

Make basic optimizations,
    meaning calculate things like sigmoid derivatives and so on before iterating over all weights and biases
    that way, each value is only calculated once, rather than every time it needs to happen
Add a helper method for calling it in an easier way, should only need to send the training data
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
inputs = [1, 2, 3, 4]
net.feedInputs(inputs)
net.calculate()

print("\nBefore:\n")
print(net.getText())

"""
gradient = net.backpropagate(2, [.5, .2], [])

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

for i in range(10):
    net.feedInputs(inputs)
    net.calculate()
    gradient = net.backpropagate(len(netSize) - 1, [.5, .2], [])
    net.applyGradient(gradient)

net.feedInputs(inputs)
net.calculate()
print("\nAfter:\n")
print(net.getText())
