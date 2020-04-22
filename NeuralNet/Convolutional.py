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
        self.decider = MatrixNetwork([sizes[-1][0], 50, self.outputs])

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
                current.append(activation(Settings.ACTIVATION_FUNC, n))

        # take outputs of final layer and feed them into a normal MatrixNetwork to get the final outputs
        decideInput = [np.max(c) for c in current]
        return self.decider.calculateInputs(decideInput)

    def applyGradient(self, gradient, dataSize):
        """
        Apply the given gradient to the weights and biases of the main and convolutional layer
        :param gradient: The gradient to apply
        :param dataSize: The size of data in the gradient
        """
        size = len(self.biases)
        for i in range(size):
            if Settings.LEARNING_RATE_BY_LAYER > 0:
                factor = float(size - i) / float(size)
            elif Settings.LEARNING_RATE_BY_LAYER < 0:
                factor = float(i + 1) / float(size)
            else:
                factor = 1
            # calculate the base value for the new weight
            w = self.weights[i] * (1 - Settings.REGULARIZATION_CONSTANT / dataSize)\
                              - gradient[0][i] * factor
            # apply the weight shrinking for the new weight
            w = np.where(w > 0, w - Settings.WEIGHT_SHRINK, w + Settings.WEIGHT_SHRINK)
            self.weights[i] = w

            self.biases[i] = self.biases[i] - gradient[1][i] * factor

            # calculate the base value for the new weight
            w = self.decider.weights[i] * (1 - Settings.REGULARIZATION_CONSTANT / dataSize)\
                              - gradient[2][i] * factor
            # apply the weight shrinking for the new weight
            w = np.where(w > 0, w - Settings.WEIGHT_SHRINK, w + Settings.WEIGHT_SHRINK)
            self.decider.weights[i] = w

            self.decider.biases[i] = self.decider.biases[i] - gradient[3][i] * factor

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
        '''
        
        potentially useful links:
        https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199
        https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509
        
        Return a thing for the weights and biases between layers
        Also need a matrix for each of the filters
        
        What are the partial derivatives when using the filters?
        
        Find activations for the entire convolutional layer filters?
        
        '''

        #
        # find activations and zActivations for the convolutional network
        #

        # find the images for the first layer based on the one input image
        current = [self.runFilter(inputs, fil) for fil in self.layers[0]]

        # initialize the list of activations with the activations from the first layer
        cActivations = [current]
        # initialize a list to store all the zActivations
        czActivations = []

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
                current.append(n)
            czActivations.append(current)
            cActivations.append([activation(Settings.ACTIVATION_FUNC, c) for c in current])

        # convert the activations into single values
        czActivations = [np.array([np.sum(a) for a in c]) for c in czActivations]
        cActivations = [np.array([np.sum(a) for a in c]) for c in cActivations]

        #
        # find activations and zActivations for the decider network
        #

        # set up lists for the weight and bias gradients
        # both of these create a numpy array of the same size as the weights and biases for that layer
        wGradient = [np.zeros(w.shape) for w in self.decider.weights]
        bGradient = [np.zeros(b.shape) for b in self.decider.biases]

        # determine the activations and zActivations of each layer
        # take the input list and ensure it's a numpy array
        dInputs = np.asarray([np.sum(c) for c in cActivations[-1]])
        # initialize the list of activations with the activations from the first layer
        activations = [dInputs]
        # initialize a list to store all the zActivations
        zActivations = []

        if drop is None:
            # iterate through each pair of weights and biases for each layer
            for w, b in zip(self.decider.weights, self.decider.biases):
                # determine the zActivation array for the current layer
                z = calc_zActivation(w, b, dInputs)
                # add the zActivation array to the list
                zActivations.append(z)
                # determine the proper activation array for the current layer,
                #   which is also used in the next loop iteration
                dInputs = activation(Settings.ACTIVATION_FUNC, z)
                # add the activation array to the list
                activations.append(dInputs)
        else:
            # iterate through each pair of weights and biases for each layer
            for j, w, b, d in zip(range(len(drop)), self.decider.weights, self.decider.biases, drop):
                # determine the zActivation array for the current layer
                z = calc_zActivation(w, b, dInputs)
                # add the zActivation array to the list
                zActivations.append(z)
                # determine the proper activation array for the current layer,
                #   which is also used in the next loop iteration
                dInputs = activation(Settings.ACTIVATION_FUNC, z)
                # set the activation values to 0 for the dropped out nodes, only for hidden layers
                # only perform drop out of this is not the output layer
                if not j == len(drop) - 1:
                    # set the input value to 0 when the dropout is within the threshold
                    dInputs = np.where(d < Settings.DROP_OUT, 0, dInputs * 0.5)
                # add the activation array to the list
                activations.append(dInputs)

        #
        # Calculate the derivatives for the backward pass for the decider network
        #

        # calculate the first part of the derivatives, which will also be the bias values
        # this is the cost derivative part, based on the expected outputs,
        #   and the derivative of the activation with the zActivation
        baseDerivatives = costDerivative(activations[-1], expected,
                                         derivActivation(Settings.ACTIVATION_FUNC, zActivations[-1]),
                                         func=Settings.COST_FUNC)
        # set the last element in the bias gradient to the initial base derivative
        bGradient[-1] = baseDerivatives

        # calculate the weight derivatives based on the activations of the previous layer, and the base derivatives
        # using np.outer creates a 2D array from 2 1D arrays by multiplying each element in the first array
        #   with each element in the second array
        wGradient[-1] = np.outer(baseDerivatives, activations[-2].transpose())

        # go through each remaining layers and calculate the remaining weight and bias derivatives
        for lay in range(2, len(self.decider.biases) + 1):
            # find the derivatives of the zActivations for the current layer
            # this is the next component in the chain rule, the activation derivatives of the zActivations
            dActivations = derivActivation(Settings.ACTIVATION_FUNC, zActivations[-lay])

            # multiply the activation derivative values with the corresponding weights
            #   and derivatives from the previous layer
            # this takes the dot product of the weights going into the current layer,
            #   which is why self.weights is indexed at [-lay + 1], rather than [-lay]
            # it is then multiplied by the values in dSigs for the other part of the derivative
            baseDerivatives = calc_multDot(self.decider.weights[-lay + 1].transpose(), baseDerivatives, dActivations)

            # set the baseDerivative values to 0 for dropped out nodes
            if drop is not None and lay > 1:
                baseDerivatives = np.where(drop[-lay] < Settings.DROP_OUT, 0, baseDerivatives * 0.5)

            # set the base derivatives in the bias list
            bGradient[-lay] = baseDerivatives

            # calculate and set the derivatives for the weight matrix
            # using the same calculation as outside the loop with np.outer,
            #   determine next part of the derivatives for the weight matrix
            #   based on the baseDerivatives, and the activations of the previous layer
            wGradient[-lay] = np.outer(baseDerivatives, activations[-lay - 1].transpose())

        #
        # Calculate the derivatives for the convolutional network
        # TODO need to calculate derivatives for the filters also, somehow
        #

        # set up lists for the weight and bias gradients
        # both of these create a numpy array of the same size as the weights and biases for that layer
        cwGradient = [np.zeros(w.shape) for w in self.weights]
        cbGradient = [np.zeros(b.shape) for b in self.biases]

        # get the dActivations for the part of the derivatives for the input nodes of the decider network
        dActivations = derivActivation(Settings.ACTIVATION_FUNC, czActivations[-1])

        # get the base derivatives for the derivatives for the input nodes of the decider network
        baseDerivatives = calc_multDot(self.decider.weights[0].transpose(), baseDerivatives, dActivations)

        # the biases for the convolutional network will be the same as the base derivatives
        cbGradient[-1] = baseDerivatives

        # calculate the weight derivatives based on the activations of the previous layer, and the base derivatives
        # using np.outer creates a 2D array from 2 1D arrays by multiplying each element in the first array
        #   with each element in the second array
        cwGradient[-1] = np.outer(baseDerivatives, cActivations[-2].transpose())

        # go through each remaining layers and calculate the remaining weight and bias derivatives
        for lay in range(2, len(self.biases) + 1):
            # find the derivatives of the zActivations for the current layer
            # this is the next component in the chain rule, the activation derivatives of the zActivations
            dActivations = derivActivation(Settings.ACTIVATION_FUNC, czActivations[-lay])
            # dActivations = derivActivation(Settings.ACTIVATION_FUNC, czActivations[-lay])

            # multiply the activation derivative values with the corresponding weights
            #   and derivatives from the previous layer
            # this takes the dot product of the weights going into the current layer,
            #   which is why self.weights is indexed at [-lay + 1], rather than [-lay]
            # it is then multiplied by the values in dSigs for the other part of the derivative
            baseDerivatives = calc_multDot(self.weights[-lay + 1].transpose(), baseDerivatives, dActivations)

            # set the base derivatives in the bias list
            cbGradient[-lay] = baseDerivatives

            # calculate and set the derivatives for the weight matrix
            # using the same calculation as outside the loop with np.outer,
            #   determine next part of the derivatives for the weight matrix
            #   based on the baseDerivatives, and the activations of the previous layer
            cwGradient[-lay] = np.outer(baseDerivatives, cActivations[-lay - 1].transpose())

        # return all the of the gradient data
        return cwGradient, cbGradient, wGradient, bGradient

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
                # variable to keep track of all the gradient values
                gradientTotal = [
                    [np.zeros(w.shape) for w in self.weights],
                    [np.zeros(b.shape) for b in self.biases],
                    [np.zeros(w.shape) for w in self.decider.weights],
                    [np.zeros(b.shape) for b in self.decider.biases]
                ]

                # go through each piece of data in the subset
                for d in dat:
                    # calculate the gradient
                    gradient = self.backpropagate(d[0], d[1], drop=None)

                    # determine scheduled learning rate
                    rate = np.power(t + 1, learnSchedule)

                    # add the gradient from the current backpropagation call to the total gradient
                    # accounting for averages and learning rate
                    for j in range(4):
                        for i in range(len(gradient[j])):
                            gradientTotal[j][i] += gradient[j][i] * rate * Settings.NET_PROPAGATION_RATE / len(dat)

                # apply the final gradient
                self.applyGradient(gradientTotal, len(data))

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
