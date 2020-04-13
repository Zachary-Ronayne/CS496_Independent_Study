from unittest import TestCase
from ImageManip.TrainingData import *


@DeprecationWarning
class TestDataSubSets(TestCase):

    def test_dataSubSet(self):
        data = [1, 2, 3, 4]

        split = dataSubSet(data, 1)
        self.assertEqual(split, [[1], [2], [3], [4]], "Testing splitting 4 elements into 1")

        split = dataSubSet(data, 2)
        self.assertEqual(split, [[1, 2], [3, 4]], "Testing splitting 4 elements into 2")

        split = dataSubSet(data, 3)
        self.assertEqual(split, [[1, 2, 3], [4]], "Testing splitting 4 elements into 3")

        split = dataSubSet(data, 4)
        self.assertEqual(split, [[1, 2, 3, 4]], "Testing splitting 4 elements into 4")

        split = dataSubSet(data, 5)
        self.assertEqual(split, [[1, 2, 3, 4]], "Testing splitting 4 elements into 5")
