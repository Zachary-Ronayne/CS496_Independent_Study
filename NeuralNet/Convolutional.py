"""

Data:
    Lists of the filters in each layer
    Will all filters be the same size?
    Will all filters in each layer be the same size?

Functions:
    Calculate:

        Take image, run it through each of the filters in the first layer, then repeat for the final layer
        Should the input size of each layer be the same as the output layer?
        The final layer of filters has connections from each value of their images, and each of those feed
            into the final output image


    Train: take data representing a gray scale image and it's corresponding color image
        or a list of images, and perform backpropagation on the network



"""

from NeuralNet.FeedForward import *

import numpy as np

import random


class Filter:

    def __init__(self, size):
        """
        Create a new filter
        :param size: Must be an integer, the radius of the filter, will always be a square.
            The final length of the square is twice the given size plus 1
        """
        s = size * 2 + 1
        self.size = size
        self.weights = np.random.rand(s, s)


class Convolution:

    def __init__(self, sizes, outputs, shrink=True):
        """
        Create a new convolutional neural network
        :param sizes: A 1D list of 2-tuples, the sizes to use for the network. The number of entries is the
            number of filter layers.
            The first index is the number of filters, the second index is the filter size.
            Size of a filter is the radius, so the square length will be twice the size plus 1.
        :param outputs: The number of outputs in the final network
        :param shrink: True if the edge pixels should be removed from the output, False otherwise, default True
        """
        self.sizes = sizes
        self.outputs = outputs
        self.shrink = shrink

        # the layers of filters
        self.layers = [[Filter(xx[1])] * xx[0] for xx in sizes]

        # the weights connecting filter layers
        self.weights = [np.random.rand(xx[0], y[0]) for xx, y in zip(sizes[1:], sizes[:-1])]

        # the biases for connecting filter layers
        self.biases = [np.random.rand(xx[0]) for xx in sizes[1:]]

        # the MatrixNetwork for converting the final layer to the individual outputs
        # TODO how do I know how many input nodes it has, just the number of matrix nodes in the final output?
        self.decider = MatrixNetwork([sizes[-1][0], self.outputs])

    def calculateInputs(self, inputs):
        """
        Take data and return the correct output based on the network
        :param inputs: The input data, should be in a 2D list, representing pixels of an image
        :return: The data, in a 1D list, of the output data
        """

        # find the images for the first layer based on the one input layer
        current = [self.runFilter(inputs, fil) for fil in self.layers[0]]

        # calculate through each layer
        for lay, bias, weights, in zip(self.layers[1:], self.biases, self.weights):
            currentLay = current
            current = []
            # go through each filter for each image, and apply all the filters
            for fil, b, weight in zip(lay, bias, weights):
                nextCurrent = []
                # go through each image
                for img in currentLay:
                    nextCurrent.append(self.runFilter(img, fil))
                # calculate the image for the current filter
                n = np.full(nextCurrent[0].shape, b)
                for c, w in zip(nextCurrent, weight):
                    n += c * w
                current.append(sigmoid(n))

        # take outputs of final layer and feed them into a normal MatrixNetwork to get the final outputs
        decideInput = [np.sum(c) for c in current]
        return self.decider.calculateInputs(decideInput)

    def applyGradient(self, gradient):
        # TODO
        """
        :param gradient: The gradient to apply
        """

    def backpropagate(self, inputs, expected, drop=None):
        """
        Calculate the gradient for changing the weights and biases.
        :param inputs: The inputs numerical values, must be a list of the same size as the input layer
        :param expected: The expected output numerical values, must be a list of the same size as the output layer
        :param drop: The array used for determining what nodes should be dropped out. Use None to disable dropout.
            Default: None
        :return: A tuple of the changed to be made to weights and biases.
            The first entry is a list of all the weight changes
            The second entry is a list of all the bias changes
        """
        # TODO
        '''
        '''
        return []

    def train(self, data, shuffle=False, split=1, times=1, func=None, learnSchedule=0):
        """
        Take the given data train the Network with it.
        :param data: The training data. Can be a single tuple containing the input and then the outputs
            Also could be a list of tuples for training data.
        :param shuffle: True to shuffle data each time the training loops, False otherwise
        :param split: Split the training data into subsets of this size.
        :param times: Train on the data this number of times
        :param func: A function that will be called each time a training interval is finished. Mostly for output.
            Must accept two int params. First is the training times count, the second is the split count
        :param learnSchedule: The degree to which learning rate will change as each training time happens.
            Use 0 to disable, negative numbers to make learning rate decrease each training time, and
            positive numbers to make learning rate increase each training time.
            Rates are based exponentially. Default 0
        """

        # if the data is not already a list of lists, put it in a list
        if isinstance(data, tuple):
            data = [data]

        # split data into subsets
        data = dataSubSet(data, split)

        # train times number of times
        for t in range(times):
            # shuffle data if applicable
            if shuffle:
                random.shuffle(data)
            # go through each subset of data of data
            for s, dat in enumerate(data):
                # TODO modify this to work for convolution
                # variable to keep track of all the gradient values
                gradientTotal = None

                # go through each piece of data in the subset
                for d in dat:
                    # calculate the gradient
                    gradient = self.backpropagate(d[0], d[1], drop=None)

                    # determine scheduled learning rate
                    rate = np.power(t + 1, learnSchedule)

                    # add the gradient from the current backpropagation call to the total gradient
                    # accounting for averages and learning rate
                    # for i in range(len(gradient[0])):
                        # gradientTotal[0][i] += gradient[0][i] * rate * Settings.NET_PROPAGATION_RATE / len(dat)
                        # gradientTotal[1][i] += gradient[1][i] * rate * Settings.NET_PROPAGATION_RATE / len(dat)

                # apply the final gradient
                self.applyGradient(gradientTotal)

                # call the extra function
                if func is not None:
                    func(t, s)

    def runFilter(self, img, filt):
        """
        Run a filter over an image to determine it's output, and return the image
        :param img: The 2D array of single values that represent the image
        :param filt: The filter object to use
        :return: The new image based on the filter
        """

        s = filt.size
        fw, fh = filt.weights.shape
        w, h = img.shape
        if self.shrink:
            outImg = np.zeros((w - s * 2, h - s * 2))
            nw, nh = outImg.shape
            for x in range(nw):
                for y in range(nh):
                    tot = 0
                    for i in range(fw):
                        for j in range(fh):
                            outImg[x, y] += img[x + s, y + s] * filt.weights[i, j]
                            tot += filt.weights[i, j]
                    outImg[x, y] /= tot
        else:
            # TODO handle the case where not resizing
            outImg = np.zeros((w, h))

        return outImg

    def random(self):
        """
        Randomize every value in this Network
        """
        for fil in self.weights:
            for i in range(len(fil)):
                for j in range(len(fil[0])):
                    fil[i, j] = random.uniform(-Settings.NET_MAX_WEIGHT, Settings.NET_MAX_WEIGHT)

        for k, w in enumerate(self.weights):
            for i in range(len(w)):
                self.biases[k][i] = random.uniform(-Settings.NET_MAX_BIAS, Settings.NET_MAX_BIAS)
                for j in range(len(w[0])):
                    w[i, j] = random.uniform(-Settings.NET_MAX_WEIGHT, Settings.NET_MAX_WEIGHT)
        self.decider.random()
