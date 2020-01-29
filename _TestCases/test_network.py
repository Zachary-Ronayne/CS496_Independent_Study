from unittest import TestCase
import NeuralNet.FeedForward as Net


# a helper start up function for getting the default state of a Network
def initTest():
    return Net.Network([1, 2])


class TestNetwork(TestCase):

    def test_calculate(self):
        net = initTest()
        pass

    def test_getOutputs(self):
        pass

    def test_feedInputs(self):
        pass

    def test_random(self):
        pass

    def test_display(self):
        pass

