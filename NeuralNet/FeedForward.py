# a file that handles all of the objects needed for a feed forward neural network

import Settings

import math
import time
import random


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
            outs.append(n.value)

        return outs

    # give the values for the inputs of the Network
    # inputs: the values to input to the Network, must be the same size as the length of the first layer
    def feedInputs(self, inputs):
        for i in range(len(inputs)):
            self.layers[0].nodes[i].value = inputs[i]

    # uss backpropagation to find the gradient vector for the given layer, using the given expected values, at the
    #   given layer
    # returns the final gradient to apply to all weights and biases
    #   the order is as follows:
    #       first element is a 2D list of the adjustments that need to be made to the weights feeding to output nodes
    #       second element is a list of all the adjustments that need to be made to the biases of the output nodes
    #       continue this with each layer, the input layer shouldn't be effected, it doesn't care about weights
    # layerI: the number of layers from the output layer to apply the back propagation algorithm
    # expected: the desired values of the given layer
    # gradient: the list of values keeping track of the adjustments that should be made to all weights and biases
    #   should give this an empty list when calling this initially
    def backpropagate(self, layerI, expected, gradient):

        if layerI == 0:
            return gradient

        wGradient = []
        for k in range(self.layers[layerI - 1].size()):
            wGradient.append([])
            for j in range(self.layers[layerI].size()):
                node0 = self.layers[layerI].nodes[j]
                node1 = self.layers[layerI - 1].nodes[k]

                change = node1.value * dSigmoid(node0.bias + node0.connections[k].weight * node1.value) * \
                    2 * (node0.value - expected[j])

                wGradient[-1].append(change)

        gradient.append(wGradient)

        bGradient = []
        for j in range(self.layers[layerI].size()):
            node0 = self.layers[layerI].nodes[j]
            total = node0.bias
            for k in range(self.layers[layerI - 1].size()):
                total += self.layers[layerI].nodes[j].connections[k].weight * \
                    self.layers[layerI - 1].nodes[k].value
            bGradient.append(dSigmoid(total) * 2 * (node0.value - expected[j]))

        gradient.append(bGradient)

        aGradient = []
        for k in range(self.layers[layerI - 1].size()):
            total = 0
            for j in range(self.layers[layerI].size()):
                w = self.layers[layerI].nodes[j].connections[k].weight
                total += w * dSigmoid(
                    self.layers[layerI].nodes[j].bias + self.layers[layerI - 1].nodes[k].value * w
                ) * 2 * (self.layers[layerI].nodes[j].value - expected[j])
            aGradient.append(total + self.layers[layerI - 1].nodes[k].value)

        return self.backpropagate(layerI - 1, aGradient, gradient)

    # apply a gradient for backpropagation to this Network, the rules for the gradient are the same as the gradient
    # returned from backpropagate()
    def applyGradient(self, gradient):
        for i in range(len(self.layers) - 1, 1, -1):
            for j in range(self.layers[i].size()):
                for k in range(self.layers[i - 1].size()):
                    self.layers[i].nodes[j].connections[k].weight -= gradient[-i * 2][k][j]

                self.layers[i].nodes[j].bias -= gradient[1 - i * 2][j]

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
        self.value = 0

    # determine the value of this Node by giving it a layer of nodes. The value is stored in this Node, and returned
    # the number of nodes in layer must be equal to the number of connections in this Node
    def feedLayer(self, layer):
        self.value = self.bias
        for i in range(layer.size()):
            self.value += layer.nodes[i].value * self.connections[i].weight

        self.value = sigmoid(self.value)

        return self.value

    # randomly generate a value for every connection in the Node and its bias
    def random(self):
        seed()
        self.bias = random.uniform(-Settings.NET_MAX_BIAS, Settings.NET_MAX_BIAS)

        for c in self.connections:
            c.random()

    # get a text representation of this node
    def getText(self):
        s = "Node: value: " + str(self.value) + ", bias:" + str(self.bias) + ", "
        for c in self.connections:
            s += c.getText()
        return s + "\n"


class Connection:

    # weight: the weight value of this connection
    def __init__(self, weight=0):
        # initialize weight, the weight of this connection combined with the values of previous nodes
        self.weight = weight

    # randomly generate a value for the weight of this connection
    def random(self):
        seed()
        self.weight = random.uniform(-Settings.NET_MAX_WEIGHT, Settings.NET_MAX_WEIGHT)

    # get a text representation of this connection
    def getText(self):
        return "weight:" + str(self.weight) + ", "


# get the value of the mathematical function sigmoid for x, return values are always in the range (0, 1)
def sigmoid(x):
    return 1.0 / (1.0 + pow(math.e, -x))


# get the value of the derivative of the mathematical function sigmoid for x
def dSigmoid(x):
    sig = sigmoid(x)
    return sig - sig * sig


# utility function to seed the random number generator
def seed():
    random.seed(time.time() + random.uniform(0, 1))
