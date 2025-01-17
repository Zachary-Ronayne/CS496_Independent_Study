# a file that handles all of the objects needed for a feed forward neural network

import Settings

from ImageManip.TrainingData import dataSubSet

import time
import random
from os import mkdir
from os.path import isdir

import numpy as np


activationList = [
    lambda x: sigmoid(x),
    lambda x: tanh(x),
    lambda x: relu(x)
]
activationDerivList = [
    lambda x: derivSigmoid(x),
    lambda x: derivTanh(x),
    lambda x: derivRelu(x)
]


class Network:
    """
    This is a proof of concept class used to demonstrate how NeuralNetworks work in an object oriented way.
    This should not be used in any real computations, use MatrixNetwork instead.
    """

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
            derivative = costDerivative(n.activation, expected[j], n.zActivation, func=Settings.COST_FUNC)
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
            # first, find the derivative activation values for each of the activations in the current layer
            # this is for the next part of calculating this layer's derivatives

            # set up a list to keep track of the activation values
            dActivation = []
            # iterate through each of the nodes in the current layer, and take it's activation through the activation
            for n in self.layers[-lay].nodes:
                dActivation.append(derivActivation(Settings.ACTIVATION_FUNC, n.zActivation))

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

                # the final derivative is the current derivative activation value times the total from the previous
                #   matrix calculation
                derivative = dActivation[j] * total
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
        createSaves()
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

        # the value of the activation before it's sent into the activation function
        self.zActivation = 0

    # determine the value of this Node by giving it a layer of nodes. The value is stored in this Node, and returned
    # the number of nodes in layer must be equal to the number of connections in this Node
    def feedLayer(self, layer):
        self.zActivation = self.bias
        for i in range(layer.size()):
            self.zActivation += layer.nodes[i].activation * self.connections[i].weight

        self.activation = activation(Settings.ACTIVATION_FUNC, self.zActivation)

        return self.activation

    # randomly generate a value for every connection in the Node and its bias
    def random(self):
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
            # multiply the weight matrix by the current values of the layer, plus the bias, put in activation function
            outputs = activation(Settings.ACTIVATION_FUNC, calc_zActivation(w, b, outputs))

        return outputs

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

        if drop is None:
            # iterate through each pair of weights and biases for each layer
            for w, b in zip(self.weights, self.biases):
                # determine the zActivation array for the current layer
                z = calc_zActivation(w, b, inputs)
                # add the zActivation array to the list
                zActivations.append(z)
                # determine the proper activation array for the current layer,
                #   which is also used in the next loop iteration
                inputs = activation(Settings.ACTIVATION_FUNC, z)
                # add the activation array to the list
                activations.append(inputs)
        else:
            # iterate through each pair of weights and biases for each layer
            for j, w, b, d in zip(range(len(drop)), self.weights, self.biases, drop):
                # determine the zActivation array for the current layer
                z = calc_zActivation(w, b, inputs)
                # add the zActivation array to the list
                zActivations.append(z)
                # determine the proper activation array for the current layer,
                #   which is also used in the next loop iteration
                inputs = activation(Settings.ACTIVATION_FUNC, z)
                # set the activation values to 0 for the dropped out nodes, only for hidden layers
                # only perform drop out of this is not the output layer
                if not j == len(drop) - 1:
                    # set the input value to 0 when the dropout is within the threshold
                    inputs = np.where(d < Settings.DROP_OUT, 0, inputs * 0.5)
                # add the activation array to the list
                activations.append(inputs)

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
        for lay in range(2, len(self.biases) + 1):
            # find the derivatives of the zActivations for the current layer
            # this is the next component in the chain rule, the activation derivatives of the zActivations
            dActivations = derivActivation(Settings.ACTIVATION_FUNC, zActivations[-lay])

            # multiply the activation derivative values with the corresponding weights
            #   and derivatives from the previous layer
            # this takes the dot product of the weights going into the current layer,
            #   which is why self.weights is indexed at [-lay + 1], rather than [-lay]
            # it is then multiplied by the values in dSigs for the other part of the derivative
            baseDerivatives = calc_multDot(self.weights[-lay + 1].transpose(), baseDerivatives, dActivations)

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

        # return the final tuple of weight and bias gradients
        return wGradient, bGradient

    def applyGradient(self, gradient, dataSize):
        """
        Apply the given gradient to the weights and biases
        :param gradient: The gradient to apply
        :param dataSize: The size of data in the gradient
        """
        # go through all the bias and weight changes, and apply them,
        # decreasing the change as they get closer to the front layer.
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
                gradientTotal = (
                    [np.zeros(w.shape) for w in self.weights],
                    [np.zeros(b.shape) for b in self.biases]
                )

                # determine the dropout array
                if Settings.DROP_OUT is None:
                    drop = None
                else:
                    # create array of nodes for dropout
                    drop = []
                    for siz in self.sizes[1:]:
                        # add random values for each of the nodes that should be dropped out
                        drop.append(np.random.rand(siz))

                # go through each piece of data in the subset
                for d in dat:
                    # calculate the gradient
                    gradient = self.backpropagate(d[0], d[1], drop=drop)

                    # determine scheduled learning rate
                    rate = np.power(t + 1, learnSchedule)

                    # add the gradient from the current backpropagation call to the total gradient
                    # accounting for averages and learning rate
                    for i in range(len(gradient[0])):
                        gradientTotal[0][i] += gradient[0][i] * rate * Settings.NET_PROPAGATION_RATE / len(dat)
                        gradientTotal[1][i] += gradient[1][i] * rate * Settings.NET_PROPAGATION_RATE / len(dat)

                # apply the final gradient
                self.applyGradient(gradientTotal, len(dat))

                # call the extra function
                if func is not None:
                    func(t, s)

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
                self.biases[k][i] = np.float64(random.uniform(-Settings.NET_MAX_BIAS, Settings.NET_MAX_BIAS))
                # go through each weight going into each specific node
                for j in range(len(self.weights[k][i])):
                    # randomly change the weight
                    self.weights[k][i, j] = np.float64(random.uniform(
                        -Settings.NET_MAX_WEIGHT, Settings.NET_MAX_WEIGHT))

    def save(self, name):
        """
        Save this network with the given name, relative to saves.
        This save file will not be compatible with the regular Network
        :param name: The name to save under. Don't include a file extension.
        """
        createSaves()
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


def createSaves():
    """
    Create the directory for the saves folder if one doesn't exist
    """
    if not isdir("saves"):
        mkdir("saves")


def activation(func, x):
    """
    Get the activation value of the given function.
    :param func: The function, use constants in Settings: SIGMOID, TANH, RELU
    :param x: The value to process
    :return: the result of the activation function
    """
    return activationList[func](x)


def derivActivation(func, x):
    """
    Get the dserivative of the activation value of the given function.
    :param func: The function, use constants in Settings: SIGMOID, TANH, RELU
    :param x: The value to process
    :return: the result of the derivative of the activation function
    """
    return activationDerivList[func](x)


def sigmoid(x):
    """
    Get the value of the mathematical function sigmoid for x,
    :param x: The value to take the sigmoid of
    :return: The sigmoid of x, always in the range (0, 1)
    """

    # make sure all values are within valid range for sigmoid to not overflow
    #   this is to ensure that, if the value of np.power(np.e, -x), no overflow happens, and if it does,
    #   then the expected value would be close enough to 0 or 1 to simply be that value
    np.clip(x, a_min=-700, a_max=700, out=x)

    # return the result
    return 1.0 / (1.0 + np.power(np.e, -x))


def derivSigmoid(x):
    """
    Get the value of the derivative of the mathematical function sigmoid for x.
    :param x: The value to take the sigmoid derivative of
    :return: The derivative of the sigmoid function at x
    """
    sig = sigmoid(x)
    return sig - sig * sig


def tanh(x):
    """
    Get the tanh value of x
    :param x: The value to take the tanh of
    :return: The tanh value
    """
    return np.tanh(x)


def derivTanh(x):
    """
    Get the derivative of the tanh value of x
    :param x: The value to take the derivative tanh of
    :return: The tanh value derivative
    """
    tan = tanh(x)
    return 1 - tan * tan


def relu(x):
    """
    Get the relu value of x
    :param x: The value to take the relu of
    :return: The relu of x
    """
    return np.maximum(x, 0, x)


def derivRelu(x):
    """
    Get the derivative of the relu of x
    :param x: The value to take the derivative of relu of
    :return: The relu derivative of x
    """
    return np.where(x > 0, 1, 0)


def costDerivative(actual, expected, zActivation, func="quadratic"):
    """
    Determine the cost for the given value and the expected value
    :param actual: The value calculated
    :param expected: The value desired
    :param zActivation: The zActivation value associated with the given actual and expected values
    :param func: The type of cost function to use for backpropagation.
        Note that the names are based on their associated cost functions, but this method calculates their derivatives,
        not the actual cost.
        Valid values: quadratic, entropy.
        Default quadratic
    :return: The cost
    """
    if func == "quadratic":
        return 2 * calc_subMult(actual, expected, zActivation)
    elif func == "entropy":
        return calc_sub(actual, expected)
    else:
        raise Exception("Invalid func type \"" + str(func) + "\"\n"
                        "Valid types: quadratic, entropy")


def averageList(values, count):
    """
    Utility function for taking the average of all values by the given count.
    Take all the values in the list.
    If the value is a number, divide it my count.
    If the value is a list, recursively call this method.
    :param values: The list of values to take the average of
    :param count: The total number of elements, used for taking the average
    """
    for i in range(len(values)):
        if isinstance(values[i], list):
            averageList(values[i], count)
        else:
            values[i] /= count


def combineList(list1, list2):
    """
    Utility function to combine numbers in lists of the same size.
    Combine the numbers in the lists.

    :param list1: If the values in this are a list, then recursively call this function.
        If the values are a number, add the values in list2 to this list.
    :param list2: The second list
    """
    for i in range(len(list1)):
        if isinstance(list1[i], list):
            combineList(list1[i], list2[i])
        else:
            list1[i] += list2[i]


def makeImageNetwork(inSize, outSize, hidden, matrixNet=True):
    """
    Create and return a Network made for taking input and output images of the specified size
    :param inSize: A tuple of (width, height) for the pixel size of input images
    :param outSize: A tuple of (width, height) for the pixel size of output images
    :param hidden: The list of numbers of nodes in hidden layers
    :param matrixNet: True to create a Matrix network, False to create an object oriented one, default True
    :return: The corresponding Network
    """
    hidden.insert(0, inSize[0] * inSize[1])
    hidden.append(outSize[0] * outSize[1] * 3)

    if matrixNet:
        return MatrixNetwork(hidden)
    else:
        return Network(hidden)


def calc_zActivation(w, b, a):
    """
    Calculate the zActivation values for the given weights biases, and activations.
    It is assumed all inputs are of valid sizes of numpy arrays
    :param w: The weights
    :param b: The biases
    :param a: The activations feeding into the next layer
    :return: The result
    """
    return np.dot(w, a) + b


def calc_multDot(a, b, c):
    """
    Find the dot product of a and b multiplied by c
    :param a: The first part of the dot product
    :param b: The second part of the dot product
    :param c: The part to multiply
    :return:
    """
    return np.dot(a, b) * c


# TODO calc method for cost derivatives
def calc_sub(a, b):
    """
    Calculate a minus b
    :param a: The a value
    :param b: The b value
    :return: The result of a minus b
    """
    return a - b


def calc_subMult(a, b, c):
    """
    Calculate a minus b all times c
    :param a: The a value
    :param b: The b value
    :param c: The c value
    :return: The final value of a minus b all times c
    """
    return (a - b) * c


# TODO calc method for sigmoid


# TODO calc method for dSigmoid


# TODO calc method for dot?


# TODO calc method for outer?
