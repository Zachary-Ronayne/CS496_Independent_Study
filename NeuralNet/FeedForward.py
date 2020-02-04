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

    # use backpropagation to find the gradient vector for the given layer, using the given expected values, at the
    #   given layer
    # before calling this method, should feed inputs and calculate data for one training example,
    #   the expected values for the initial call of this method should be that of the training example
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
        # base case, when layerI is 0, that means there are no more layers to propagate through
        if layerI == 0:
            return gradient

        # find the values of the individual activations going into sigmoid functions, only for this layer
        # set up variable for activations
        activations = []
        # iterate through the previous layer
        for k in range(self.layers[layerI - 1].size()):
            # add one more list to the activations list
            activations.append([])
            # iterate through the current layer
            for j in range(self.layers[layerI].size()):
                # get the node currently being iterated over
                node0 = self.layers[layerI].nodes[j]
                # get the node that has the connection from the current iterated connection,
                #   and where that connection feeds into the currently iterated node
                node1 = self.layers[layerI - 1].nodes[k]
                # add the activation
                activations[-1].append(node0.bias + node0.connections[k].weight * node1.value)

        # the list that will contain all the changes made to weights
        wGradient = []
        # iterate through the number of connections feeding into the current layer
        for k in range(self.layers[layerI - 1].size()):
            # add one list to the weights
            wGradient.append([])
            # iterate through the number of nodes in the current layer
            for j in range(self.layers[layerI].size()):
                # get the node currently being iterated over
                node0 = self.layers[layerI].nodes[j]
                # get the node that has the connection from the current iterated connection,
                #   and where that connection feeds into the currently iterated node
                node1 = self.layers[layerI - 1].nodes[k]

                # determine change in weight,
                #   this is the derivative of the cost function with respect to the current connection's weight
                # there are 3 terms, one for each part of the chain rule
                # the first term is the derivative of the value plugged into the sigmoid, the activation,
                #   with respect to the connection's weight
                # the second term is the derivative of the sigmoid function,
                #   with respect to the value, the activation, plugged into the sigmoid
                # the third term is the derivative of the cost function,
                #   with respect to the output of the sigmoid, the activation
                change = node1.value * dSigmoid(activations[k][j]) * 2 * (node0.value - expected[j])

                # add the change for the weight onto the list of weights, placing it at the end
                wGradient[-1].append(change)

        # add the 2D list of weights as one more element to the gradient
        gradient.append(wGradient)

        # the list that will contain all the values for all the changes in biases
        bGradient = []
        # iterate through every node in the current layer
        for j in range(self.layers[layerI].size()):
            # get the node currently being iterated through
            node0 = self.layers[layerI].nodes[j]
            # add the change in bias value to the bias gradient list
            # this means finding the derivative of the cost function with respect to the bias
            # the first term is not here, because it works out to just 1, having no effect on the derivative
            # the second term is the derivative of the sigmoid with the value feeding into it
            # the third term is the derivative of the cost function with respect to the activation
            bGradient.append(dSigmoid(node0.activation) * 2 * (node0.value - expected[j]))

        # add the bias gradient list to the main gradient list
        gradient.append(bGradient)

        # the list keeping track of the changes to activation
        #   these will not be change values, as they are the desired values for the activations
        #   this means the computed value for the derivative will be added to the actual activation
        aGradient = []
        # iterate through all the nodes in the previous layer
        for k in range(self.layers[layerI - 1].size()):
            # initialize counter for value fed into the derivative for finding change in activation
            # this doesn't need an initial bias, as it's not finding an activation total,
            #   but the total change to the cost function
            total = 0
            # iterate though all the nodes in the current layer
            for j in range(self.layers[layerI].size()):
                # get the node of the current layer
                node = self.layers[layerI].nodes[j]
                # find the weight of the connection feeding from the previous layer to the current layer
                w = node.connections[k].weight
                # find the derivative of the cost function with respect to the activation of the previous layer
                # the first term is the weight feeding into the current node
                # the second term is the derivative of the sigmoid with respect to the activation of current layer
                # the third term is the derivative of the cost function,
                #   with respect to the activation of the current node
                total += w * dSigmoid(activations[k][j]) * 2 * (node.value - expected[j])
            # change the total to be the average change in the cost function for all expected nodes
            total /= len(expected)
            # add the value of the change in activation with the actual activation of the node from the previous layer
            aGradient.append(total + self.layers[layerI - 1].nodes[k].value)

        # recursively call this algorithm
        # backpropagate on the previous layer,
        #   use the values from the desired activations for the expected values
        #   use the current gradient to ensure that all weight and bias change values are stored in order
        return self.backpropagate(layerI - 1, aGradient, gradient)

    # apply a gradient for backpropagation to this Network, the rules for the gradient are the same as the gradient
    # returned from backpropagate()
    def applyGradient(self, gradient):
        # count down from the last layer to the second to last layer
        for i in range(len(self.layers) - 1, 0, -1):
            # iterate through all the nodes from the current layer
            for j in range(self.layers[i].size()):
                # iterate through all the nodes from the previous layer
                for k in range(self.layers[i - 1].size()):
                    # take the connection connecting the two nodes being iterated over,
                    #   and subtract its corresponding change in weight
                    self.layers[i].nodes[j].connections[k].weight -= \
                        gradient[-i * 2][k][j] * Settings.NET_PROPAGATION_RATE

                # take the current node and subtract its corresponding change in bias
                self.layers[i].nodes[j].bias -= \
                    gradient[1 - i * 2][j] * Settings.NET_PROPAGATION_RATE

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
            gradient = self.backpropagate(len(self.layers) - 1, data[1], [])

            # if the total gradient is empty, then set it to the new gradient
            if not gradientTotal:
                gradientTotal = gradient
            # otherwise, add the new gradient to the total gradient
            else:
                # TODO loop this probably needs to be changed to be more efficient, both in terms of time and code
                for i in range(len(gradientTotal)):
                    # the even entries are 2D weight lists, need to iterate through all values
                    if i % 2 == 0:
                        for j in range(len(gradientTotal[i])):
                            for k in range(len(gradientTotal[i][j])):
                                gradientTotal[i][j][k] += gradient[i][j][k]
                    # the odd entries are a single list, need to iterate through one list
                    else:
                        for j in range(len(gradientTotal[i])):
                            gradientTotal[i][j] += gradient[i][j]

        # take the average values for the gradients
        for i in range(len(gradientTotal)):
            # the even entries are 2D weight lists, need to iterate through all values
            if i % 2 == 0:
                for j in range(len(gradientTotal[i])):
                    for k in range(len(gradientTotal[i][j])):
                        gradientTotal[i][j][k] /= len(trainingData)
            # the odd entries are a single list, need to iterate through one list
            else:
                for j in range(len(gradientTotal[i])):
                    gradientTotal[i][j] /= len(trainingData)

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

        # the value of the activation before it's sent into the sigmoid function
        self.activation = 0

    # determine the value of this Node by giving it a layer of nodes. The value is stored in this Node, and returned
    # the number of nodes in layer must be equal to the number of connections in this Node
    def feedLayer(self, layer):
        self.activation = self.bias
        for i in range(layer.size()):
            self.activation += layer.nodes[i].value * self.connections[i].weight

        self.value = sigmoid(self.activation)

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
