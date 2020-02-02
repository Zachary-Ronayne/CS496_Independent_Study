"""

TODO

Image manipulation

Add settings for selecting a range of a video file, time codes and percentages
Add settings for selecting every certain number of frames
Make GUI and commandline program for image manipulation

"""


import NeuralNet.FeedForward as Net

net = Net.Network([4, 6, 2])
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

for i in range(1000):
    net.feedInputs(inputs)
    net.calculate()
    gradient = net.backpropagate(2, [.5, .2], [])
    net.applyGradient(gradient)

net.feedInputs(inputs)
net.calculate()
print("\nAfter:\n")
print(net.getText())
