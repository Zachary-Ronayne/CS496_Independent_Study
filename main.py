"""

TODO


Test cases for Feed Forward


Image manipulation

Add settings for selecting a range of a video file, time codes and percentages
Add settings for selecting every certain number of frames
Make GUI and commandline program for image manipulation

"""


import NeuralNet.FeedForward as Net

net = Net.Network([4, 5, 6, 2])
net.calculate()
net.display()

print()
print("-------------------------------------------------------------------------------------------------------------")
print()

net.random()
print("\n\nBefore calc:")
net.display()

net.calculate()
print("\n\nAfter calc:")
net.display()

net.feedInputs([1, 2, 3, 4])
net.calculate()
print("\n\nSet values calc:")
net.display()
