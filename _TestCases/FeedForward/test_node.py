from unittest import TestCase
import NeuralNet.FeedForward as Net
import _TestCases.FeedForward.test_connection as TestCon

import os


@DeprecationWarning
class TestNode(TestCase):
    def test_init(self):
        n = Net.Node()
        self.assertTrue(n.bias == 0, "Default bias should be 0, was: " + str(n.bias))
        self.assertTrue(n.connections == [], "Default connections should be empty, was: " + str(n.connections))
        self.assertTrue(n.activation == 0, "Default value should be 0")

        n = Net.Node(None)
        self.assertTrue(n.connections == [], "None connections should be empty, was: " + str(n.connections))

        n = Net.Node(connections=[])
        self.assertTrue(n.connections == [], "Empty list connections should be empty, was: " + str(n.connections))

        cons = [Net.Connection(1), Net.Connection(2)]
        n = Net.Node(bias=2, connections=cons)
        self.assertTrue(n.connections == cons, "List with 2 connections should be " + str(cons) +
                        ", was:" + str(n.connections))
        self.assertTrue(n.bias == 2, "Given bias should be 2, was: " + str(n.bias))

        n = Net.Node(connections=3)
        self.assertTrue(len(n.connections) == 3, "Node should have 3 connections")

    def test_feedLayer(self):
        n = Net.Node(bias=1, connections=[Net.Connection(1), Net.Connection(-2)])
        nodes = [Net.Node(), Net.Node()]
        nodes[0].activation = 1.2
        nodes[1].activation = .8
        layer = Net.Layer(nodes)
        n.feedLayer(layer)
        expect = .64565630622
        self.assertAlmostEqual(n.activation, expect, 6, "After calculating node, it's value should be about " + str(expect) +
                               ", was: " + str(n.activation))

    def test_random(self):
        n = Net.Node(connections=4)
        n.random()
        testRandom(self, n)

    def test_getText(self):
        n = Net.Node(bias=1.1, connections=[Net.Connection(1.3), Net.Connection(-2.2)])
        n.activation = 2.5
        s = n.getText()
        expect = "Node: value: 2.5, bias:1.1, weight:1.3, weight:-2.2, \n"
        self.assertTrue(s == expect, "Text of node should be \n\"" + expect + "\" was \n\"" + s + "\"")

    def test_save(self):
        n = Net.Node(bias=2.1, connections=[Net.Connection(1), Net.Connection(-.5)])
        n.activation = .3
        n.zActivation = .2
        with open("nodeSaveTest.txt", "w") as f:
            n.save(f)

        loadN = Net.Node()
        with open("nodeSaveTest.txt", "r") as f:
            loadN.load(f)

        self.assertEqual(n.activation, loadN.activation, "After saving and loading this node, the value should be .3 was: " +
                         str(loadN.activation))
        self.assertEqual(n.zActivation, loadN.zActivation, "After saving and loading this node,"
                                                         "the activation should be .2 was: " +
                         str(loadN.zActivation))
        self.assertEqual(n.bias, loadN.bias, "After saving and loading this node, the bias should be .2 was: " +
                         str(loadN.bias))
        self.assertEqual(len(loadN.connections), 2, "After saving and loading this node,"
                                                    "there should be 2 connections, found: " +
                         str(len(loadN.connections)))
        self.assertEqual(loadN.connections[0].weight, 1, "After saving and loading this node,"
                                                         "the first connection should be 1, was: " +
                         str(loadN.connections[0].weight))
        self.assertEqual(loadN.connections[1].weight, -.5, "After saving and loading this node,"
                                                           "the second connection should be -.5, was: " +
                         str(loadN.connections[1].weight))

        os.remove("nodeSaveTest.txt")


def testRandom(test, node):
    test.assertFalse(node.bias == 0, "Random bias should not be default bias of 0")
    for c in node.connections:
        TestCon.testRandom(test, c)