from unittest import TestCase
import NeuralNet.FeedForward as Net
import _TestCases.FeedForward.test_node as TestNode


class TestLayer(TestCase):
    def test_init(self):
        lay = Net.Layer()
        self.assertTrue(lay.nodes == [], "By default, nodes should be empty, was: " + str(lay.nodes))

        lay = Net.Layer([])
        self.assertTrue(lay.nodes == [], "When created with an empty list, nodes should be empty,"
                                         "was: " + str(lay.nodes))

        lay = Net.Layer(None)
        self.assertTrue(lay.nodes == [], "None nodes should be empty, was: " + str(lay.nodes))

        lay = Net.Layer((2, 3))
        length = len(lay.nodes)
        self.assertTrue(length == 2, "The layer should have 2 nodes, had: " + str(length))
        length = len(lay.nodes[0].connections)
        self.assertTrue(length == 3, "Each node should have 3 connections, node 0 had: " + str(length))
        length = len(lay.nodes[1].connections)
        self.assertTrue(length == 3, "Each node should have 3 connections, node 1 had: " + str(length))

        nodes = [Net.Node(1), Net.Node(2)]
        lay = Net.Layer(nodes)
        self.assertTrue(lay.nodes == nodes, "Nodes should match: " + str(nodes) + "was: " + str(lay.nodes))

    def test_size(self):
        lay = Net.Layer((5, 0))
        self.assertTrue(lay.size() == 5, "Size of the layer should be 5, was: " + str(lay.size))

    def test_calculate(self):
        lay = Net.Layer((3, 2))
        for n in lay.nodes:
            self.assertTrue(n.value == 0, "Before calculations, value should be 0, was: " + str(n.value))

        inLay = Net.Layer((2, 0))
        inLay.random()
        inLay.nodes[0].value = .4
        inLay.nodes[1].value = -.5
        lay.calculate(inLay)
        for n in lay.nodes:
            self.assertFalse(n.value == 0, "After calculations, value should not be 0")

    def test_random(self):
        lay = Net.Layer((4, 2))
        lay.random()
        testRandom(self, lay)

    def test_getText(self):
        lay = Net.Layer((3, 4))
        s = lay.getText()
        expect = "Layer:\n" \
                 "Node: value: 0, bias:0, weight:0, weight:0, weight:0, weight:0, \n" \
                 "Node: value: 0, bias:0, weight:0, weight:0, weight:0, weight:0, \n" \
                 "Node: value: 0, bias:0, weight:0, weight:0, weight:0, weight:0, \n"
        self.assertTrue(expect == s, "Text of layer should be:\n" + expect + "\nwas:\n" + s)


def testRandom(test, lay):
    for n in lay.nodes:
        TestNode.testRandom(test, n)