import numpy as np
import util
import sys
import math
import matplotlib.pyplot as plt

### NOTE : You need to complete logreg implementation first!
class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        m = []
        for i in range(len(x)):
            vec = x[i]
            m.append(vec)
        matrix = np.array(m)
        multd = np.matmul(np.transpose(matrix), matrix)
        ident = np.identity(multd.shape[0])
        res = np.linalg.solve(multd, ident)

        multd = np.matmul(np.transpose(matrix), y)
        self.theta = np.dot(res, multd)
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        lst = []
        for i in x:
            exp = math.e ** (self.theta.dot(i))
            exp += 1
            lst.append(1/exp)
        return np.array(lst)
        # *** END CODE HERE ***


# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***
    # Part (a): Train and test on true labels
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()

    plt.figure()

    t, x, y = loadData(test_path)
    zero = []
    one = []
    minx = int(min(x[:,0]))
    maxx = int(max(x[:,0]))
    miny = int(min(x[:,1]))
    maxy = int(max(x[:,1]))
    for i in range(len(x)):
        if t[i] == 0.0:
            zero.append(x[i])
        else:
            one.append(x[i])

    zero = np.array(zero)
    one = np.array(one)

    plt.scatter(zero[:,0], zero[:,1])
    plt.scatter(one[:,0], one[:,1])
    plt.xlabel("X_1")
    plt.xticks(np.arange(minx-2, maxx + 2, (maxx-minx)/10))
    plt.yticks(np.arange(miny-2, maxy+2, (maxy - miny)/10))
    plt.ylabel("x_2")
    test_model = LogisticRegression()
    test_model.fit(x, y)

    z = x[:,0]

    xset = [i for i in range(int(min(z)) - 1,int(max(z)) * 2)]
    yset = test_model.predict(xset)
    np.savetxt(output_path_true, yset)
    plt.plot(xset, yset)
    # Part (b): Train on y-labels and test on true labels
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    # Part (f): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to output_path_adjusted
    plt.show()
    # *** END CODER HERE


def loadData(path):
    f = open(path)
    f.readline()

    ts = []
    xs = []
    ys = []
    for line in f:
        lst = line.strip().split(",")
        ts.append(float(lst[0]))
        x1 = float(lst[1])
        x2 = float(lst[2])
        xs.append(np.array([x1, x2]))
        ys.append(float(lst[3]))
    f.close()
    return (ts, np.array(xs), ys)



if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
