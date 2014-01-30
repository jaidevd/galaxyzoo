import numpy as np


X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=np.float)
yxor = np.array([0, 1, 1, 0], dtype=np.float)
yand = np.zeros((4,),dtype=np.float)
yand[3] = 1


class Perceptron(object):

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass


class MultiLayerPerceptron(object):

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def fit(self, x, y):
        pass

    def predict(self, y):
        pass


if __name__ == "__main__":
    mlp = MultiLayerPerceptron()

