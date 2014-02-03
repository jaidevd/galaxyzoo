import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=np.float)
yxor = np.array([0, 1, 1, 0], dtype=np.float)
yand = np.zeros((4,),dtype=np.float)
yand[3] = 1


def simulate_2d_gaussian_clusters():
    cov = [[0.1, 0],[0, 0.1]]
    a = np.random.multivariate_normal([1, 1], cov, 500)
    b = np.random.multivariate_normal([-1, -1], cov, 500)
    a = np.concatenate((a, np.ones((500,1))), axis=1)
    b = np.concatenate((b, -np.ones((500,1))), axis=1)
    x = np.concatenate((a, b), axis=0)
    np.random.shuffle(x)
    return x


class Perceptron(object):

    def __init__(self, alpha=0.3, theta=0.2, maxiter=1000):
        self.alpha = alpha
        self.theta = theta
        self.maxiter = maxiter

    def fit(self, x, y, learning_steps=True, weights=None, bias=None):

        if weights is None:
            self.org_weights = np.random.random((x.shape[1],))
            self.weights = self.org_weights.copy()
        elif weights.ndim > 1:
            self.weights = weights.ravel()
        else:
            self.weights = weights

        if bias is None:
            self.org_bias = np.random.random()
            self.bias = self.org_bias

        self.x = x

        if y.ndim > 1:
            self.y = y.ravel()
        else:
            self.y = y

        prediction = self.evaluate(self.x)
        n_iter = 0

        steps = (50, 100, 500)
        learning = [(self.org_weights[0], self.org_weights[1],
                     self.org_bias)]

        while not np.allclose(prediction, self.y):
            addition = self.alpha * np.sum(self.y * self.x.T, axis=1)
            self.weights += addition
            self.bias += self.alpha * np.sum(self.y)
            prediction = self.evaluate(self.x)
            n_iter += 1
            #if n_iter % 100 == 0:
            #    print n_iter
            if n_iter in steps:
                learning.append((self.weights[0], self.weights[1], self.bias))
            print n_iter
            if n_iter == self.maxiter:
                break

        learning.append((self.weights[0], self.weights[1], self.bias))
        if learning_steps:
            return learning, n_iter




    def evaluate(self, x):
        y_in = np.dot(self.x, self.weights) + self.bias
        y = np.zeros(self.y.shape)
        try:
            y[y_in > self.theta] = 1
            y[y_in < -self.theta] = -1
        except ValueError:
            from IPython.core.debugger import Tracer;Tracer()()
        return y

    def predict(self, x):
        y = np.dot(x, self.weights) + self.bias
        y[y > self.theta] = 1
        y[y < self.theta] = -1
        return y





class MultiLayerPerceptron(object):

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def fit(self, x, y):
        pass

    def predict(self, y):
        pass


if __name__ == "__main__":
    x = simulate_2d_gaussian_clusters()
    x = x - x.mean(0)
    x = x/x.std(0)
    trainx, trainy = x[:700,:2], x[:700,2]
    testx, testy = x[700:,:2], x[700:,2]
    perc = Perceptron(theta=0, maxiter=10000)
    perc.fit(trainx, trainy)
    pred = perc.predict(testx)
    print np.allclose(pred, testy)

