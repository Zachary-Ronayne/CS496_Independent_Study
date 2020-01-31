from unittest import TestCase
import NeuralNet.FeedForward as Net
import _TestCases.FeedForward.test_layer as TestLayer


class TestNetwork(TestCase):
    def test_init(self):
        net = Net.Network()
        self.assertTrue(net.layers == [], "With default network, layers should be empty list, was: " + str(net.layers))

        net = Net.Network([])
        self.assertTrue(net.layers == [], "With empty list network, layers should be empty list,"
                                          " was: " + str(net.layers))

        net = Net.Network(None)
        self.assertTrue(net.layers == [], "With none layers, layers should be empty list, was: " + str(net.layers))

    def test_calculate(self):
        layers = []

        nodes = [Net.Node([]), Net.Node([])]
        nodes[0].value = 3.3
        nodes[1].value = -1.5
        layers.append(Net.Layer(nodes))

        nodes = [
            Net.Node(bias=.2, connections=[Net.Connection(1), Net.Connection(-1)]),
            Net.Node(bias=-1.3, connections=[Net.Connection(.5), Net.Connection(.2)])
        ]
        layers.append(Net.Layer(nodes))

        nodes = [
            Net.Node(bias=-.8, connections=[Net.Connection(1.5), Net.Connection(.2)]),
            Net.Node(bias=.45, connections=[Net.Connection(4), Net.Connection(3)])
        ]
        layers.append(Net.Layer(nodes))

        net = Net.Network(layers)
        net.calculate()

        expect = [0.688359339593154, 0.9974285761888321]
        actual = net.getOutputs()
        for i in range(2):
            self.assertAlmostEqual(expect[i], actual[i], 6,
                                   "Network output should be: " + str(expect[i]) +
                                   " was: " + str(actual[i]))

    def test_getOutputs(self):
        net = Net.Network([1, 4])
        net.layers[-1].nodes[0].value = 3
        net.layers[-1].nodes[1].value = -2
        net.layers[-1].nodes[2].value = 2
        net.layers[-1].nodes[3].value = 7
        self.assertTrue(net.getOutputs() == [3, -2, 2, 7], "Outputs should be: [3, -2, 2, 7],"
                                                           " was: " + str(net.getOutputs()))

    def test_feedInputs(self):
        net = Net.Network([4, 2])
        expect = [1, 2, 5, -3]
        net.feedInputs(expect)
        for i in range(4):
            v = net.layers[0].nodes[i].value
            self.assertTrue(v == expect[i], "Input node value should be " + str(expect[i]) + ", was: " + str(v))

    def test_random(self):
        net = Net.Network([2, 3, 4])
        net.random()
        for lay in net.layers:
            TestLayer.testRandom(self, lay)

    def test_getText(self):
        net = Net.Network([3, 4, 2])
        s = net.getText()
        expect = "Network:\n" \
                 "Layer:\n" \
                 "Node: value: 0, bias:0, \n" \
                 "Node: value: 0, bias:0, \n" \
                 "Node: value: 0, bias:0, \n" \
                 "Layer:\n" \
                 "Node: value: 0, bias:0, weight:0, weight:0, weight:0, \n" \
                 "Node: value: 0, bias:0, weight:0, weight:0, weight:0, \n" \
                 "Node: value: 0, bias:0, weight:0, weight:0, weight:0, \n" \
                 "Node: value: 0, bias:0, weight:0, weight:0, weight:0, \n" \
                 "Layer:\n" \
                 "Node: value: 0, bias:0, weight:0, weight:0, weight:0, weight:0, \n" \
                 "Node: value: 0, bias:0, weight:0, weight:0, weight:0, weight:0, \n"

        self.assertTrue(s == expect, "Text from network should be:\n" + expect + "was:\n" + s)
