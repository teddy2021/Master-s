import util
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pdb
np.seterr(all='raise')

factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        m = []
        for i in range(0, len(X)):
            vec = X[i]
            m.append(vec)
        matrix = np.array(m)
        multd = np.matmul(np.transpose(matrix), matrix)
        ident = np.identity(multd.shape[0])
        res = np.linalg.solve(multd, ident)

        multd = np.matmul(np.transpose(matrix), y)
        self.theta = np.dot(res, multd)

        # *** END CODE HERE ***

    def fit_GD(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the gradient descent algorithm.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        alpha = 0.01

        res = self.predict(X)

        update = np.zeros(np.shape(X)[1],dtype='float')

        for i in range(0, len(X)):
            step = res[i]
            step -= y[i]
            step *= alpha
            step *= X[i]
            update += step

        update /= len(X)
        self.theta -= update


        # *** END CODE HERE ***

    def fit_SGD(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the stochastic gradient descent algorithm.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        alpha = 0.01

        for i in range(0, len(X)):
            vec = X[i]
            res = self.predict(vec)
            val = alpha * (y[i] - res)*vec
            self.theta -= val
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        if k == 1:
            return np.array([[1,x[1]] for x in X])
        res = []
        for x in X:
            val = x[1]
            lst = []
            for i in range(k + 1):
                lst.append(val ** i)
            res.append(np.array(lst))
        return np.array(res)
        # *** END CODE HERE ***

    def create_cosine(self, k, X):
        """
        Generates a cosine with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        res = []
        for x in X:
            val = x[1]
            i = 0
            lst = []
            for i in range(k+1):
                lst.append(val**i)
            lst.append(math.cos(val))
            res.append(np.array(lst))
        return np.array(res)

        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        if self.theta is None:
            return None
        if np.ndim(X) == 1:
            return self.theta.dot(X)

        results = []
        for x in X:
            results.append(self.theta.dot(x))
        return np.array(results).reshape(len(results), )
        # *** END CODE HERE **ee if array has specific shape


def run_exp(train_path, cosine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.pdf'):

    train_x, train_y = util.load_dataset(train_path, add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-0.1, 1.1, 1000)
    plt.figure()

    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        model = LinearModel(np.zeros([1, k + 1]))
        if not cosine:
            x = model.create_poly(k, train_x)
        else:
            x = model.create_cosine(k, train_x)

        if "Normal" in filename:
            f_type = "Normal"
            model.fit(x, train_y)
        elif "SGD" in filename:
            f_type = "Stochastic Gradient Descent"
            for i in range(3):
                for j in range(10000):
                    model.fit_SGD(x, train_y)
        else:
            f_type = "Gradient Descent"
            for i in range(3):
                for j in range(10000):
                    model.fit_GD(x, train_y)

        if cosine:
            plot_y = model.predict(model.create_cosine(k, plot_x))
        else:
            plot_y = model.predict(model.create_poly(k, plot_x))
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2.5, 2.5)
        plt.plot(plot_x[:, 1], plot_y,
                label='k={:d}, fit={:s}'.format(k, f_type))

    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()


def main(medium_path, small_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***

    run_exp(medium_path, False, [3], "MedNorm")
    run_exp(medium_path, False, [3], "MedGD")
    run_exp(medium_path, False, [3], 'MedSGD')
    run_exp(medium_path, False, [3,5,10,20], "MedNormalFitChange")
    run_exp(medium_path, True, [3,5,10,20], 'MedNormalCosFitChange')
    run_exp(small_path, False, [1,3,5,10,20], 'SmallNormalOverfit')
    run_exp(small_path, True, [1,3,5,10,20], 'SmallNormalCosOverfit')

    # *** END CODE HERE ***


if __name__ == '__main__':
    main(medium_path='medium.csv',
         small_path='small.csv')
