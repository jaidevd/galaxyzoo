import unittest
from nose import SkipTest

import numpy as np

from galaxyzoo.ml.mlp import Perceptron
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA


class TestPerceptron(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        iris_data = load_iris()
        data = iris_data['data']
        data = PCA(2).fit_transform(data)
        data = np.concatenate((data, iris_data['target'].reshape((150,1))),
                              axis=1)
        np.random.shuffle(data)
        data[data == 0] = -1
        cls.data = data
        cls.classifier = Perceptron(theta=0)

    def test_classify_setosa_vs_virginica(self):
        raise SkipTest()
        data = self.data[self.data[:,2] != 2, :]
        trainx = data[:50,:2]
        trainy = data[:50,2]
        testx = data[50:,:2]
        testy = data[50:,2]
        self.classifier.fit(trainx, trainy.ravel())
        prediction = self.classifier.predict(testx)
        self.assertTrue(np.allclose(prediction, testy.ravel()))

    def test_linear_inseparable_classes(self):
        x = np.random.multivariate_normal([0, 0], [[1, 0],[0, 1]], 1000)
        targets = np.ones((1000,))
        targets[:500] = -1
        np.random.shuffle(targets)
        learning, n_iter = self.classifier.fit(x, targets)
        self.assertEqual(n_iter, 1000)


if __name__ == "__main__":
    unittest.main()
