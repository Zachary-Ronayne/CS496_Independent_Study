# a file that handles all of the objects needed for a feed forward neural network

import Settings

from ImageManip.TrainingData import dataSubSet

import math
import time
import random

import numpy as np


class Network:

    # initialize a feed forward network
    # layers: use None or no parameter for an empty network
    #   use a list of layers to give the Network with those layers
    #   use a list of integers to create a network with layers where each layer has that number of nodes
    def __init__(self, layers=None):
        # initialize layers, make it an empty list if no variable is provided
        # layers is all of the layers of nodes in this Network
        if layers is None or layers == []:
            self.layers = []

        # if the list contains numbers, initialize each layer to the number of nodes
        elif isinstance(layers[0], int):
            self.layers = [Layer((layers[0], 0))]
            for i in range(1, len(layers)):
                self.layers.append(Layer((layers[i], layers[i - 1])))

        # otherwise use the list directly as given
        else:
            self.layers = layers

    # calculate the values of the outputs of this network
    def calculate(self):
        for i in range(1, len(self.layers)):
            self.layers[i].calculate(self.layers[i - 1])

    # get a list of all the values of all the output nodes in this Network, will always be the length of the last layer
    def getOutputs(self):
        outs = []
        for n in self.layers[-1].nodes:
            outs.append(n.activation)

        return outs

    # give the values for the inputs of the Network
    # inputs: the values to input to the Network, must be the same size as the length of the first layer
    def feedInputs(self, inputs):
        for i in range(len(inputs)):
            self.layers[0].nodes[i].activation = inputs[i]

    def calculateInputs(self, inputs):
        """
        Utility method to calculate and return the values of the Network
        :param inputs: The values to feed into the network
        :return:
        """
        self.feedInputs(inputs)
        self.calculate()
        return self.getOutputs()

    # use backpropagation to find the gradient vector for the given layer, using the given expected values, at the
    #   given layer
    # before calling this method, should call feedInputs for one training example,
    #   the expected values for the initial call of this method should be that of the training example
    # returns the final gradient to apply to all weights and biases
    #   this is in the form of a tuple
    #   the 0 index is is the weights
    #   the 1 index is the biases
    #   the weights are a 3D list, each element is a 2D list of the weights for each layer in reverse order
    #       so the first element is the weights from the output layer to the second to last layer
    #       and the last element is the weights from the input layer to the second layer
    #   the biases are a 2D list, each element is the biases for each layer in reverse order
    #       so the first element is the biases for the last layer
    #       and the last element is the biases for the first layer
    # expected: the desired values of the output layer
    def backpropagate(self, expected):
        # calculate activations and values of nodes, these can directly be accessed from the Network
        self.calculate()

        # the list that will contain all the changes to be made to weights
        wGradient = []

        # the list that will contain all the values for all the changes in biases
        # also create add an empty list for the first layer
        bGradient = [[]]

        # calculate the partial parts of the derivatives, the part that is the same for all terms
        #   this only happens on the last layer
        baseDerivatives = []
        for j, n in enumerate(self.layers[-1].nodes):
            # find the derivative value for the first 2 terms
            derivative = dSigmoid(n.zActivation) * costDerivative(n.activation, expected[j])
            # add the derivative to the baseDerivatives list
            baseDerivatives.append(derivative)
            # the bias values get the same derivative, so add it to that list also
            bGradient[-1].append(derivative)

        # now set up the weight change values for the last layer
        # first add an empty list for the 2D list of the weights for the last layer
        wGradient.append([])
        # iterate through all nodes in the last layer
        for k in range(len(self.layers[-1].nodes)):
            # add an empty list for the next set of weights
            wGradient[-1].append([])
            # iterate through all nodes in the second to last layer
            for node in self.layers[-2].nodes:
                # calculate the change in weight for the connection between the two current nodes
                wGradient[-1][-1].append(baseDerivatives[k] * node.activation)

        # now, iterate through the rest of the layers, applying the derivatives as the go back through the layers
        # the indexes for the layers use the negative index of lay, to go backwards through the list
        for lay in range(2, len(self.layers)):
            # first, find the derivative sigmoid values for each of the activations in the current layer
            # this is for the next part of calculating this layer's derivatives

            # set up a list to keep track of the sigmoid values
            dSigmoids = []
            # iterate through each of the nodes in the current layer, and take it's activation through the sigmoid
            for n in self.layers[-lay].nodes:
                dSigmoids.append(dSigmoid(n.zActivation))

            # now, take those values and multiply them with corresponding weights, and the derivatives from the
            #   previous layer to determine the values for the biases
            # add one entry to the list for the bias gradient
            bGradient.append([])
            # make a new list to store all the new derivatives
            newDerivatives = []

            # iterate through each of the nodes in the current layer
            for j in range(len(self.layers[-lay].nodes)):
                # find the derivative value for the first 2 terms
                # add up the values as the previous weight matrix
                total = 0
                for i in range(len(baseDerivatives)):
                    total += baseDerivatives[i] * self.layers[-lay + 1].nodes[i].connections[j].weight

                # the final derivative is the current derivative sigmoid value times the total from the previous
                #   matrix calculation
                derivative = dSigmoids[j] * total
                # set the new base derivative
                newDerivatives.append(derivative)

                # store the change in bias in the bias gradient for the nodes in the current layer to the derivatives
                #   calculated in the previous step
                # the bias values get the same derivative, so add it to that list also
                bGradient[-1].append(derivative)

            # set the base derivative list to the newly calculated list
            baseDerivatives = newDerivatives

            # find the derivatives for the weights, and store them in the weight gradient

            # add a new entry to the weight gradient
            wGradient.append([])

            # iterate through all the nodes in the current layer
            for k in range(len(self.layers[-lay].nodes)):
                # add an empty list for the next set of weights
                wGradient[-1].append([])
                # iterate through all nodes in the layer before the current layer
                for node in self.layers[-lay - 1].nodes:
                    # calculate the change in weight for the connection between the two current nodes
                    wGradient[-1][-1].append(baseDerivatives[k] * node.activation)

        # return a tuple of the weight and bias gradients
        return wGradient, bGradient

    # apply a gradient for backpropagation to this Network, the rules for the gradient are the same as the gradient
    # returned from backpropagate()
    def applyGradient(self, gradient):
        # count down from the last layer, using -i as the index, this does not include the first layer
        for i in range(1, len(self.layers)):
            # iterate through all the nodes from the current layer
            for j in range(self.layers[-i].size()):
                # iterate through all the nodes from the previous layer
                for k in range(self.layers[-i - 1].size()):
                    # apply each weight change
                    self.layers[-i].nodes[j].connections[k].weight -= \
                        gradient[0][i - 1][j][k] * Settings.NET_PROPAGATION_RATE

                # apply the bias changes
                self.layers[-i].nodes[j].bias -= gradient[1][i - 1][j] * Settings.NET_PROPAGATION_RATE

    # train the network on training data
    # send a single tuple of two lists for one input, send a list of tuples for multiple pieces of data
    # the tuple should be of the form (input, expectedOutput)
    def train(self, trainingData):
        # if the data is not already a list of lists, put it in a list
        if isinstance(trainingData, tuple):
            trainingData = [trainingData]

        # variable to keep track of all the gradient values
        gradientTotal = []

        # for each piece of data, train on it
        for data in trainingData:
            # send the inputs and calculate the data
            self.feedInputs(data[0])
            self.calculate()
            # calculate the gradient
            gradient = self.backpropagate(data[1])

            # if the total gradient is empty, then set it to the new gradient
            if not gradientTotal:
                gradientTotal = gradient
            # otherwise, add the new gradient to the total gradient
            else:
                combineList(gradientTotal, gradient)

        # take the average values for the gradients
        averageList(gradientTotal, len(trainingData))

        # apply the final gradient
        self.applyGradient(gradientTotal)

    # randomly generate a value for every weight and bias in the Network
    def random(self):
        for l in self.layers:
            l.random()

    # get a text representation of this network
    def getText(self):
        s = "Network:\n"
        for l in self.layers:
            s += l.getText()

        return s

    # save this Network with the given file name relative to the saves folder
    def save(self, name):
        # open the file
        with open("saves/" + name + ".txt", "w") as f:
            # save the number of layers
            f.write(str(len(self.layers)) + "\n")

            # save each layer
            for l in self.layers:
                l.save(f)

    # load this Network with the given file name relative to the saves folder
    def load(self, name):
        # open the file
        with open("saves/" + name + ".txt", "r") as f:
            # get the number of layers
            size = int(f.readline())
            self.layers = []
            # load each layer
            for i in range(size):
                lay = Layer()
                lay.load(f)
                self.layers.append(lay)

    def convertToMatrix(self):
        """
        Take the weights and biases in this Network and convert them into a MatrixNetwork
        :return: The converted Matrix Network
        """
        sizes = [len(lay.nodes) for lay in self.layers]
        newNet = MatrixNetwork(sizes)
        for k, lay in enumerate(self.layers[1:]):
            for j, n in enumerate(lay.nodes):
                newNet.biases[k][j] = n.bias
                for i, c in enumerate(n.connections):
                    newNet.weights[k][j][i] = c.weight

        return newNet


class Layer:

    # nodes: don't include or use None to give an empty list of nodes
    #   use a list of nodes to give the layer that many nodes
    #   use a tuple in the form (m, n), to give this Layer m nodes, where each node has n connections
    def __init__(self, nodes=None):
        # initialize nodes, make it an empty list if no variable is provided
        # nodes is the list of nodes in this layer
        if nodes is None or nodes == []:
            self.nodes = []
        # if nodes contains a tuple, then it should contain integers for the number of nodes and connections per node
        elif isinstance(nodes, tuple):
            self.nodes = []
            for i in range(nodes[0]):
                self.nodes.append(Node(connections=nodes[1]))
        else:
            self.nodes = nodes

    # get the number of nodes in this Layer
    def size(self):
        return len(self.nodes)

    # calculate the values of all the nodes in this layer by feeding in the given layer
    def calculate(self, layer):
        for n in self.nodes:
            n.feedLayer(layer)

    # randomly generate a value for every weight and bias in the Layer
    def random(self):
        for n in self.nodes:
            n.random()

    # get a text representation of this layer
    def getText(self):
        s = "Layer:\n"
        for n in self.nodes:
            s += n.getText()
        return s

    # save this Layer to the given file IO object in write mode
    def save(self, f):
        # save the number of nodes
        f.write(str(self.size()) + "\n")
        # save each node
        for n in self.nodes:
            n.save(f)

    # load this Layer with the given file IO object in read mode
    def load(self, f):
        # load the number of nodes
        size = int(f.readline())
        self.nodes = []
        # load each node
        for i in range(size):
            n = Node()
            n.load(f)
            self.nodes.append(n)


class Node:

    # bias: the bias value of this node
    # connections: don't include or use None for an empty list
    #   use a list of connection objects to have a list of connections feed into this Node
    #   use an integer to give this node that number of connections, all at default values
    def __init__(self, bias=0, connections=None):
        # initialize bias, the bias of this node combined with the values of all connections feeding into this node
        self.bias = bias

        # initialize connections, make it an empty list if no variable is provided
        # connections is a list of all the connections feeding into this node
        if connections is None or connections == []:
            self.connections = []
        # if connections is an integer, create a list of connections of that length
        elif isinstance(connections, int):
            self.connections = []
            for i in range(connections):
                self.connections.append(Connection())
        else:
            self.connections = connections

        # value is the current values stored in this node
        # value is always initialized to 0
        self.activation = 0

        # the value of the activation before it's sent into the sigmoid function
        self.zActivation = 0

    # determine the value of this Node by giving it a layer of nodes. The value is stored in this Node, and returned
    # the number of nodes in layer must be equal to the number of connections in this Node
    def feedLayer(self, layer):
        self.zActivation = self.bias
        for i in range(layer.size()):
            self.zActivation += layer.nodes[i].activation * self.connections[i].weight

        self.activation = sigmoid(self.zActivation)

        return self.activation

    # randomly generate a value for every connection in the Node and its bias
    def random(self):
        seed()
        self.bias = random.uniform(-Settings.NET_MAX_BIAS, Settings.NET_MAX_BIAS)

        for c in self.connections:
            c.random()

    # get a text representation of this node
    def getText(self):
        s = "Node: value: " + str(self.activation) + ", bias:" + str(self.bias) + ", "
        for c in self.connections:
            s += c.getText()
        return s + "\n"

    # save this Node to the given file IO object in write mode
    def save(self, f):
        # save data about the node
        f.write(str(len(self.connections)) + " " + str(self.bias) + " " + str(self.activation)
                + " " + str(self.zActivation) + "\n")
        # save each connection
        for c in self.connections:
            c.save(f)
        # if at least one connection was saved, make a new line
        if len(self.connections) > 0:
            f.write("\n")

    # load this node with the given file IO object in read mode
    def load(self, f):
        # load the first line containing data about this node
        line = f.readline().split(' ')
        size = int(line[0])
        self.bias = float(line[1])
        self.activation = float(line[2])
        self.zActivation = float(line[3])

        self.connections = []

        # if there is at least one connection load in all connections
        if size > 0:
            # load the line containing connection values
            line = f.readline().split(' ')

            # load each connection
            for i in range(size):
                c = Connection(float(line[i]))
                self.connections.append(c)


class Connection:

    # weight: the weight value of this connection
    def __init__(self, weight=0.0):
        # initialize weight, the weight of this connection combined with the values of previous nodes
        self.weight = weight

    # randomly generate a value for the weight of this connection
    def random(self):
        seed()
        self.weight = random.uniform(-Settings.NET_MAX_WEIGHT, Settings.NET_MAX_WEIGHT)

    # get a text representation of this connection
    def getText(self):
        return "weight:" + str(self.weight) + ", "

    # save this Connection to the given file IO object in write mode
    def save(self, f):
        # save the weight of this connection3
        f.write(str(self.weight) + " ")


class MatrixNetwork:
    """
    A class that performs the same Network operations, but using much more efficient methods for storage and processing.
    This class has less convenience features than Network, and is harder to navigate from a code perspective
    """

    def __init__(self, sizes):
        """
        Initialize the numpy arrays for this Network based on the given size.
        All weights and biases are initialized to zero
        :param sizes: a list of numbers,
            the first entry is the length of the input list,
            the last entry is the length of the output list,
            all other entries are the sizes of hidden layers.
            So [2, 3, 4, 5] has 2 inputs, then a hidden layer of 3 nodes,
            then a hidden layer of 4 nodes, then 5 output nodes
        """

        # initialize the weight and bias lists
        self.weights = []
        self.biases = []
        self.sizes = sizes

        # add appropriately sized numpy arrays
        for k in range(len(sizes) - 1):
            self.weights.append(np.zeros((sizes[k + 1], sizes[k]), np.float64))
            self.biases.append(np.zeros((sizes[k + 1]), np.float64))

    def layerCount(self):
        """
        Get the number of layers in this Network
        :return: The number of layers
        """
        return len(self.sizes)

    def layerSize(self, i):
        """
        Get the number of nodes in the ith layer
        :param i: The layer
        :return: the number of nodes
        """
        return self.sizes[i]

    def calculateInputs(self, inputs):
        """
        Take the given input values and calculate the corresponding outputs.
        :param inputs: The inputs numerical values, must be a list of the same size as the inputs
        :return: A list of values representing all the outputs of the Network
        """
        # set up outputs as a numpy array
        outputs = np.asarray(inputs)
        # iterate through each layer of weights and biases
        for w, b in zip(self.weights, self.biases):
            # multiply the weight matrix by the current values of the layer, plus the bias, sent through sigmoid
            outputs = sigmoid(np.dot(w, outputs) + b)

        return outputs

    def backpropagate(self, inputs, expected):
        """
        Calculate the gradient for changing the weights and biases.
        :param inputs: The inputs numerical values, must be a list of the same size as the input layer
        :param expected: The expected output numerical values, must be a list of the same size as the output layer
        :return: A tuple of the changed to be made to weights and biases.
            The first entry is a list of all the weight changes
            The second entry is a list of all the bias changes
        """

        # set up lists for the weight and bias gradients
        # both of these create a numpy array of the same size as the weights and biases for that layer
        wGradient = [np.zeros(w.shape) for w in self.weights]
        bGradient = [np.zeros(b.shape) for b in self.biases]

        # determine the activations and zActivations of each layer
        # take the input list and ensure it's a numpy array
        inputs = np.asarray(inputs)
        # initialize the list of activations with the activations from the first layer
        activations = [inputs]
        # initialize a list to store all the zActivations
        zActivations = []
        # iterate through each pair of weights and biases for each layer
        for w, b in zip(self.weights, self.biases):
            # determine the zActivation array for the current layer
            z = np.dot(w, inputs) + b
            # add the zActivation array to the list
            zActivations.append(z)
            # determine the proper activation array for the current layer,
            #   which is also used in the next loop iteration
            inputs = sigmoid(z)
            # add the activation array to the list
            activations.append(inputs)

        # calculate the first part of the derivatives, which will also be the bias values
        # this is the cost derivative part, based on the expected outputs,
        #   and the derivative of the sigmoid with the zActivation
        baseDerivatives = costDerivative(activations[-1], expected) * dSigmoid(zActivations[-1])
        # set the last element in the bias gradient to the initial base derivative
        bGradient[-1] = baseDerivatives

        # calculate the weight derivatives based on the activations of the previous layer, and the base derivatives
        # using np.outer creates a 2D array from 2 1D arrays by multiplying each element in the first array
        #   with each element in the second array
        wGradient[-1] = np.outer(baseDerivatives, activations[-2].transpose())

        # go through each remaining layers and calculate the remaining weight and bias derivatives
        for lay in range(2, len(self.biases) + 1):
            # find the derivatives of the zActivations for the current layer
            # this is the next component in the chain rule, the sigmoid derivatives of the zActivations
            dSigs = dSigmoid(zActivations[-lay])

            # multiply the sigmoid derivative values with the corresponding weights
            #   and derivatives from the previous layer
            # this takes the dot product of the weights going into the current layer,
            #   which is why self.weights is indexed at [-lay + 1], rather than [-lay]
            # it is then multiplied by the values in dSigs for the other part of the derivative
            baseDerivatives = np.dot(self.weights[-lay + 1].transpose(), baseDerivatives) * dSigs

            # set the base derivatives in the bias list
            bGradient[-lay] = baseDerivatives

            # calculate and set the derivatives for the weight matrix
            # using the same calculation as outside the loop with np.outer,
            #   determine next part of the derivatives for the weight matrix
            #   based on the baseDerivatives, and the activations of the previous layer
            wGradient[-lay] = np.outer(baseDerivatives, activations[-lay - 1].transpose())

        # return the final tuple of weight and bias gradients
        return wGradient, bGradient

    def applyGradient(self, gradient):
        """
        Apply the given gradient to the weights and biases
        :param gradient: The gradient to apply
        """
        # go through all the bias and weight changes, and apply them
        for i in range(len(self.biases)):
            self.weights[i] = self.weights[i] - gradient[0][i]
            self.biases[i] = self.biases[i] - gradient[1][i]

    def train(self, data, shuffle=False, split=1, times=1):
        """
        Take the given data train the Network with it.
        :param data: The training data. Can be a single tuple containing the input and then the outputs
            Also could be a list of tuples for training data.
        :param shuffle: True to shuffle data each time the training loops, False otherwise
        :param split: Split the training data into subsets of this size.
        :param times: Train on the data this number of times
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
            for dat in data:
                # variable to keep track of all the gradient values
                gradientTotal = (
                    [np.zeros(w.shape) for w in self.weights],
                    [np.zeros(b.shape) for b in self.biases]
                )
                # go through each piece of data in the subset
                for d in dat:
                    # calculate the gradient
                    gradient = self.backpropagate(d[0], d[1])

                    # add the gradient from the current backpropagation call to the total gradient
                    # accounting for averages and leanring rate
                    for i in range(len(gradient[0])):
                        gradientTotal[0][i] += gradient[0][i] * Settings.NET_PROPAGATION_RATE / len(dat)
                        gradientTotal[1][i] += gradient[1][i] * Settings.NET_PROPAGATION_RATE / len(dat)

                # apply the final gradient
                self.applyGradient(gradientTotal)

    def random(self):
        """
        set all the weights and bias values to random values based on the maximum values in Settings
        """
        # seed the random number generator
        random.seed(time.time())

        # go through each layer
        for k in range(len(self.weights)):
            # go through each weight and bias going into each node of the layer
            for i in range(len(self.biases[k])):
                # randomly change the bias
                self.biases[k][i] = random.uniform(-Settings.NET_MAX_BIAS, Settings.NET_MAX_BIAS)
                # go through each weight going into each specific node
                for j in range(len(self.weights[k][i])):
                    # randomly change the weight
                    self.weights[k][i, j] = random.uniform(-Settings.NET_MAX_WEIGHT, Settings.NET_MAX_WEIGHT)

    def save(self, name):
        """
        Save this network with the given name, relative to saves.
        This save file will not be compatible with the regular Network
        :param name: The name to save under. Don't include a file extension.
        """
        with open("saves/" + name + ".txt", "w") as f:
            # write the layer size data
            f.write(" ".join([str(s) for s in self.sizes]))
            f.write("\n")
            # go through each layer
            for k in range(len(self.weights)):
                # save the biases of this node
                f.write(" ".join([str(b) for b in self.biases[k]]))
                f.write("\n")
                # go through each node in the current layer
                for i in range(len(self.weights[k])):
                    # save the weights for that layer
                    f.write(" ".join([str(w) for w in self.weights[k][i]]))
                    f.write("\n")

    def load(self, name):
        """
        Load this network from the file with the given name, relative to saves.
        This method is not compatible with the regular Network save files
        :param name: The name of the file to load. Don't include a file extension.
        """
        with open("saves/" + name + ".txt", "r") as f:
            # get the values for the layers
            layerSizes = [int(i) for i in f.readline().split(" ")]

            # initialize the network based on the number of layers
            self.__init__(layerSizes)

            # go through all the layer sections for the weights and biases
            for k in range(len(layerSizes) - 1):
                # get the number of nodes and their connections in the given layer
                sizes = (layerSizes[k], layerSizes[k + 1])

                # load in the line containing all the bias values
                biases = f.readline().split()
                # initialize an array to load all the biases
                b = np.empty(sizes[1], dtype=np.float64)
                # set all the values
                for i in range(len(b)):
                    b[i] = biases[i]

                # add the bias array to the biases list
                self.biases[k] = b

                # initialize an array to hold all the weights
                w = np.empty(sizes, dtype=np.float64)
                # go through each dimension of the weight array
                for i in range(sizes[1]):
                    # get the next line of weights
                    weights = f.readline().split()
                    for j in range(len(weights)):
                        # set each weight values
                        w[j, i] = weights[j]
                # add the weight array to the weights list
                self.weights[k] = w.transpose()

    def getText(self):
        """
        Get a user readable string representing this Network
        :return: the string
        """
        text = []
        for i, w, b, in zip(range(len(self.weights)), self.weights, self.biases):
            text.append("".join(["Between layer ", str(i + 1), " and ", str(i + 2)]))
            text.append("Biases:")
            text.append(str(b))
            text.append("Weights:")
            text.append(str(w))
            text.append("")
        return "\n".join(text)

    def convertToNormal(self):
        """
        Take the weights and biases in this Network and convert them into a normal Network
        :return: The converted normal Network
        """
        newNet = Network(self.sizes)
        for k, lay in enumerate(newNet.layers[1:]):
            for j, n in enumerate(lay.nodes):
                n.bias = self.biases[k][j]
                for i, c in enumerate(n.connections):
                    c.weight = self.weights[k][j][i]

        return newNet


# get the value of the mathematical function sigmoid for x, return values are always in the range (0, 1)
def sigmoid(x):
    return 1.0 / (1.0 + np.power(math.e, -x))


# get the value of the derivative of the mathematical function sigmoid for x
def dSigmoid(x):
    sig = sigmoid(x)
    return sig - sig * sig


def costDerivative(actual, expected):
    return 2 * (actual - expected)


# utility function to seed the random number generator
def seed():
    random.seed(time.time() + random.uniform(0, 1))


# utility function for taking the average of all values by the given count
# take all the values in the list
# if the value is a number, divide it my count
# if the value is a list, recursively call this method
def averageList(values, count):
    for i in range(len(values)):
        if isinstance(values[i], list):
            averageList(values[i], count)
        else:
            values[i] /= count


# utility function to combine numbers in lists of the same size
# combine the numbers in the lists
# if the values in list1 are a list, then recursively call this function
# if the values are a number, combine them from list 2
def combineList(list1, list2):
    for i in range(len(list1)):
        if isinstance(list1[i], list):
            combineList(list1[i], list2[i])
        else:
            list1[i] += list2[i]


def makeImageNetwork(width, height, hidden, matrixNet=True):
    """
    Create and return a Network made for taking input and output images of the specified size
    :param width: The width of the images the Network should handle
    :param height: The height of the images the Network should handle
    :param hidden: The list of numbers of nodes in hidden layers
    :param matrixNet: True to create a Matrix network, False to create an object oriented one, default True
    :return: The corresponding Network
    """
    hidden.insert(0, width * height)
    hidden.append(width * height * 3)

    if matrixNet:
        return MatrixNetwork(hidden)
    else:
        return Network(hidden)
