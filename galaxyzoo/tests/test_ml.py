import unittest
from galaxyzoo.ml.mlp import MultiLayerPerceptron, Perceptron
import numpy as np


class TestML(unittest.TestCase):

    def setUp(self):
        self.x = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]], dtype=np.float)

    def test_perceptron_and(self):
        percept = Perceptron()
        y = np.zeros((4,), dtype=np.float)
        y[3] = 1
        percept.fit(self.x, y)
        y_hat = percept.predict(self.x)
        self.assertTrue(np.allclose(y_hat, y))

    def test_backprop_xor(self):
        """
        Test if backpropagation works for the XOR gate
        :return:
        """
        mlp = MultiLayerPerceptron()
        y = np.array([0,1,1,0])
        mlp.fit(self.x, y)
        y_hat = mlp.predict(self.x)
        self.assertTrue(np.allclose(y_hat, y))
