import numpy as np
import util
#import math


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    # *** START CODE HERE ***
    # Train a GDA classifier
    model = GDA()
    model.fit(x_train, y_train)
    # Plot decision boundary on validation set
    plot_x, plot_y = util.load_dataset(valid_path)
    util.plot(plot_x, plot_y, model.theta, save_path + ".png")
    # Use np.savetxt to save outputs from validation set to save_path
    np.savetxt(save_path, model.predict(plot_x))
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        if None == self.theta:
            self.theta = np.zeros((x.ndim + 1,))
        # Find phi, mu_0, mu_1, and sigma
        phi = y.sum() / len(y)
        n0 = 0
        d0 = 0
        n1 = 0
        d1 = 0
        for i in range(len(y)):
            if(y[i] == 0):
                n0 += x[i]
                d0 += 1
            elif(y[i] == 1):
                n1 += x[i]
                d1 += 1
        mu = []
        mu.append(n0/d0)
        mu.append(n1/d1)

        summa = np.zeros((x.ndim, x.ndim))
        for i in range(len(x)):
            vec = (x[i] - mu[ int( y[i] ) ])
            summa += np.outer(vec, vec.T)
        covariance = summa/len(y)
#        denom = (2 * Math.PI )** (x.ndim/2) * Math.sqrt(np.linalg.det(covariance))
#        pxy0 = np.exp(-0.5(x-mu[0]).T * covariance * (x-mu[0])) / denom
#        pxy1 = np.exp(-0.5(x-mu[1]).T * covariance * (x-mu[1])) / denom
        # Write theta in terms of the parameters
        self.theta[0] = 0.5 * ( mu[0] @ np.linalg.inv(covariance) @ mu[0] - mu[1] @ np.linalg.inv(covariance) @ mu[1])
        self.theta[0] -= np.log((1-phi)/phi)
        self.theta[1:] = np.linalg.inv(covariance) @ (mu[0] - mu[1]) * -1
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1/(1 + np.exp(- (self.theta[1:] * x) + self.theta[0]))
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
