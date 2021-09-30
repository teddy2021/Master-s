import util
import numpy as np
import matplotlib.pyplot as plt
import math
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
            vec = X[i][0]
            m.append(vec)
        matrix = np.array(m)

        multd = np.matmul(np.transpose(matrix), matrix)
        ident = np.identity(np.shape(multd)[0])
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

        step = alpha * np.sum(res[:] - y[:])
        for i in range(0, len(res)):
            vec = X[i]
            for j in range(0, len(self.theta[0])):
                val = self.theta[0][j]
                val -= step * vec[j]
                self.theta[0][j] = val
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

        lst = []
        for i in range(0, len(X)):
            lst.append( self.predict(X[i]))
            res = np.array(lst)
            vec = X[i]
            step = alpha * (np.sum(res[:] - y[:len(res) - 1]))
            for j in range(0, len(self.theta)):
                val = self.theta[0][i]
                val -= step * vec[j]
                self.theta[0][j] = val
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
            return np.array([x[1] for x in X])
        res = []
        for x in X:
            val = x[1]
            i = 0
            lst = []
            while i <= k:
                lst.append(val ** i)
                i += 1
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
            while i <= k:
                lst.append(val**i)
                i += 1
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
        results = []
        for x in X:
            results.append(self.theta.T[0].dot(x.T[0]))
        return np.array(results).reshape((len(results), ))
        # *** END CODE HERE ***


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
        model = LinearModel(np.zeros([1,4]))
        x = model.create_poly(3, train_x)
        exp = k
        if exp > 3:
            exp = 3
        print("10^", exp + 1, "iterations")

        for i in range(0, (100*(10**(exp)))):
            model.fit_GD(x, train_y)

        plot_y = model.predict(model.create_poly(3, plot_x))
        f_type = "Gradient Descent"
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
    run_exp(small_path, ks=[1,2,3], filename="SmallGD.pdf")
    run_exp(medium_path, ks=[1,2,3], filename="MediumGD.pdf")
    # *** END CODE HERE ***


if __name__ == '__main__':
    main(medium_path='medium.csv',
         small_path='small.csv')
