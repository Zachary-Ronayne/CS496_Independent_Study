from unittest import TestCase
from NeuralNet import FeedForward as N


@DeprecationWarning
class TestFeedForward(TestCase):
    def test_sigmoid(self):
        self.assertAlmostEqual(N.sigmoid(0.0), 0.5, 8, "Should correctly return sigmoid value")
        self.assertAlmostEqual(N.sigmoid(5.0), 0.9933071490757, 8, "Should correctly return sigmoid value")
        self.assertAlmostEqual(N.sigmoid(-5.0), 0.006692850924, 8, "Should correctly return sigmoid value")

    def test_dSigmoid(self):
        self.assertAlmostEqual(N.derivSigmoid(0.0), 0.25, 8, "Should correctly return sigmoid value")
        self.assertAlmostEqual(N.derivSigmoid(5.0), 0.0066480566707901, 8, "Should correctly return derivative of sigmoid")
        self.assertAlmostEqual(N.derivSigmoid(-5.0), 0.006648056670790, 8, "Should correctly return derivative of sigmoid")

    def test_averageList(self):
        avg = [5]
        N.averageList(avg, 2)
        self.assertEqual(avg, [2.5], "Each value in each list should be divided by 2 of the original list")

        avg = [5, [4, 7]]
        N.averageList(avg, 2)
        self.assertEqual(avg, [2.5, [2, 3.5]], "Each value in each list should be divided by 2 of the original list")

        avg = [[5, 8], [4, 7]]
        N.averageList(avg, 2)
        self.assertEqual(avg, [[2.5, 4], [2, 3.5]],
                         "Each value in each list should be divided by 2 of the original list")

    def test_combineList(self):
        lis1 = [1, 2, 3, 4]
        lis2 = [2, 3, 5, 6]
        N.combineList(lis1, lis2)
        self.assertEqual(lis1, [3, 5, 8, 10], "the values in the list should be equal to the combined values of: " +
                         str(lis1) + " and " + str(lis2))

        lis1 = [1, 2, [2, 5], 4]
        lis2 = [2, 3, [-3, 8], 6]
        N.combineList(lis1, lis2)
        self.assertEqual(lis1, [3, 5, [-1, 13], 10], "the values in the list should be equal to the combined "
                                                     "values of: " + str(lis1) + " and " + str(lis2))
