from unittest import TestCase
import NeuralNet.FeedForward as Net

import os


class TestConnection(TestCase):

    def test_init(self):
        c = Net.Connection()
        self.assertTrue(c.weight == 0, "Weight should be 0 by default, was: " + str(c.weight))
        c = Net.Connection(2)
        self.assertTrue(c.weight == 2, "Weight should be set to 2, was: " + str(c.weight))

    def test_random(self):
        c = Net.Connection(1)
        c.random()
        testRandom(self, c)

    def test_getText(self):
        c = Net.Connection(-2.1)
        s = c.getText()
        self.assertTrue(s == "weight:-2.1, ", "Incorrect formatting for text of connection, \"" + s + "\"")

    def test_save(self):
        c = Net.Connection(2.1)
        with open("connectionSaveTest.txt", "w") as f:
            c.save(f)

        with open("connectionSaveTest.txt", "r") as f:
            value = float(f.readline())

        self.assertEqual(value, c.weight, "After saving and loading this connection, the weight should be 2.1, was: " +
                         str(value))

        os.remove("connectionSaveTest.txt")


def testRandom(test, con):
    test.assertFalse(con.weight == 1, "Weight should be set to a random value, not 1, value is: " + str(con.weight))
