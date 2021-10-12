import numpy as np
import util
import sys
import math
import matplotlib.pyplot as plt
from datetime import datetime as dt
import pdb

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
        i = 0
        delta = np.Infinity
        likelihood = self.log_likelihood(x, y)

        hessian, gradient = self.derivs(x, y)

        while i < self.max_iter and not abs(delta) <= self.eps:
            update =  (np.linalg.inv(hessian) @ gradient).reshape(self.theta.shape)
            self.theta -=  update
            i += 1
            hessian, gradient = self.derivs(x, y)
            new_likelihood = self.log_likelihood(x,y)
            delta = new_likelihood - likelihood
            likelihood = new_likelihood

        # *** END CODE HERE ***


    def derivs(self, X, y):
        hypothesis = self.predict(X)
        val = [hypothesis[i] - y[i] for i in range(len(hypothesis))]
        gradient =X.T.dot(val)
        gradient /= len(X)
        val = hypothesis.T.dot(1-hypothesis)[0][0]
        ar = [val for x in range(len(X))]
        h_center = np.diag(ar)
        hessian = X.T @ h_center @ X
        hessian /= len(X)
        return hessian, gradient


    def log_likelihood(self, x, y):
        predictions = self.predict(x)
        lst = []
        for i in range(len(predictions)):
            lst.append(
                (np.log(predictions[i]) * y) + (np.log(1-predictions[i])*(1-y))
            )
        return np.sum(
            np.array(lst)
        )

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        lst = [(1/(1+np.exp(-self.theta.dot(i)))) for i in x]
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

    #plt.figure()

    #t, x, y = loadData(test_path)
    #zero = []
    #one = []
    #minx = int(min(x[:,0]))
    #maxx = int(max(x[:,0]))
    #miny = int(min(x[:,1]))
    #maxy = int(max(x[:,1]))
    #for i in range(len(x)):
    #    if t[i] == 0.0:
    #        zero.append(x[i])
    #    else:
    #        one.append(x[i])

    #zero = np.array(zero)
    #one = np.array(one)

    #plt.scatter(zero[:,0], zero[:,1], label='t=zero')
    #plt.scatter(one[:,0], one[:,1], label='t=one')
    #plt.title("Test T")
    #plt.xlabel("X_1")
    #plt.ylabel("x_2")
    #plt.xticks(np.arange(minx-2, maxx + 2, (maxx-minx)/10))
    #plt.yticks(np.arange(miny-2, maxy+2, (maxy - miny)/10))


    #test_true = LogisticRegression(theta_0=np.zeros([1,x.ndim]))
    #start = dt.now()
    #print("Starting test data fit with t values at", start)
    #test_true.fit(x, t)
    #end = dt.now()
    #print("Finished at ", end)

    #plotx = np.ones([1000, x.ndim])
    #plotx[:,0] = np.linspace(minx, maxx, 1000)
    #plotx[:,1] = np.linspace(minx, maxx, 1000)
    #yset = test_true.predict(plotx)

    #np.savetxt(output_path_true, yset)


    #plt.plot(plotx[:,1], yset, 'r-', label='Decision Boundry')
    #plt.legend()
    #plt.savefig("Q2TestWT")

    #plt.figure()

    #t, x, y = loadData(valid_path)
    #zero = []
    #one = []
    #minx = int(min(x[:,0]))
    #maxx = int(max(x[:,0]))
    #miny = int(min(x[:,1]))
    #maxy = int(max(x[:,1]))
    #for i in range(len(x)):
    #    if t[i] == 0.0:
    #        zero.append(x[i])
    #    else:
    #        one.append(x[i])

    #zero = np.array(zero)
    #one = np.array(one)

    #plt.scatter(zero[:,0], zero[:,1], label='t=zero')
    #plt.scatter(one[:,0], one[:,1], label='t=one')
    #plt.title("Valid T")
    #plt.xlabel("X_1")
    #plt.xticks(np.arange(minx-2, maxx + 2, (maxx-minx)/10))
    #plt.yticks(np.arange(miny-2, maxy+2, (maxy - miny)/10))
    #plt.ylabel("x_2")


    #valid_true = LogisticRegression(theta_0=np.zeros([1, x.ndim]))
    #start = dt.now()
    #print("Starting valid data fit with t values at ", start)
    #valid_true.fit(x, t)
    #end = dt.now()
    #print("Finished at ", end)

    #plotx = np.ones([1000, x.ndim])
    #plotx[:,0] = np.linspace(minx, miny, 1000)
    #plotx[:,1] = np.linspace(miny, maxy, 1000)
    #yset = valid_true.predict(plotx)

    #plt.plot(plotx[:,1], yset, 'r-', label='Decision Boundry')
    #plt.legend()
    #plt.savefig("Q2ValidWT")

    #plt.figure()

    #t, x, y = loadData(train_path)
    #zero = []
    #one = []
    #minx = int(min(x[:,0]))
    #maxx = int(max(x[:,0]))
    #miny = int(min(x[:,1]))
    #maxy = int(max(x[:,1]))
    #for i in range(len(x)):
    #    if t[i] == 0.0:
    #        zero.append(x[i])
    #    else:
    #        one.append(x[i])

    #zero = np.array(zero)
    #one = np.array(one)

    #plt.scatter(zero[:,0], zero[:,1], label='t=zero')
    #plt.scatter(one[:,0], one[:,1], label='t=one')
    #plt.title("Train T")
    #plt.xlabel("X_1")
    #plt.xticks(np.arange(minx-2, maxx + 2, (maxx-minx)/10))
    #plt.yticks(np.arange(miny-2, maxy+2, (maxy - miny)/10))
    #plt.ylabel("x_2")


    #train_true = LogisticRegression(theta_0=np.zeros([1, x.ndim]))
    #print("Starting true fit with t at", dt.now())
    #train_true.fit(x, t)
    #print("Ending true fit with t at", dt.now())

    #plotx = np.ones([1000, x.ndim])
    #plotx[:,0] = np.linspace(minx, miny, 1000)
    #plotx[:,1] = np.linspace(miny, maxy, 1000)
    #yset = train_true.predict(plotx)


    #plt.plot(plotx[:,1], yset, 'r-', label='Decision Boundry')
    #plt.legend()
    #plt.savefig("Q2TrainT")
    ## Part (b): Train on y-labels and test on true labels
    ## Make sure to save predicted probabilities to output_path_naive using np.savetxt()


    #plt.figure()

    #t, x, y = loadData(test_path)
    #zero = []
    #one = []
    #minx = int(min(x[:,0]))
    #maxx = int(max(x[:,0]))
    #miny = int(min(x[:,1]))
    #maxy = int(max(x[:,1]))
    #for i in range(len(x)):
    #    if t[i] == 0.0:
    #        zero.append(x[i])
    #    else:
    #        one.append(x[i])

    #zero = np.array(zero)
    #one = np.array(one)

    #plt.scatter(zero[:,0], zero[:,1], label='t=zero')
    #plt.scatter(one[:,0], one[:,1], label='t=one')
    #plt.title("Test y")
    #plt.xlabel("X_1")
    #plt.xticks(np.arange(minx-2, maxx + 2, (maxx-minx)/10))
    #plt.yticks(np.arange(miny-2, maxy+2, (maxy - miny)/10))
    #plt.ylabel("x_2")


    #test_naive = LogisticRegression(theta_0=np.zeros([1, x.ndim]))
    #print("Startaing test fit with y at", dt.now())
    #test_naive.fit(x, y)
    #print("Ending test fit with y at", dt.now())

    #plotx = np.ones([1000, x.ndim])
    #plotx[:,0] = np.linspace(minx, miny, 1000)
    #plotx[:,1] = np.linspace(miny, maxy, 1000)
    #yset = test_naive.predict(plotx)

    #np.savetxt(output_path_naive, yset)


    #plt.plot(plotx[:,1], yset, 'r-', label='Decision Boundry')
    #plt.legend()

    #plt.savefig("Q2TestWY")


    #plt.figure()

    #t, x, y = loadData(valid_path)
    #zero = []
    #one = []
    #minx = int(min(x[:,0]))
    #maxx = int(max(x[:,0]))
    #miny = int(min(x[:,1]))
    #maxy = int(max(x[:,1]))
    #for i in range(len(x)):
    #    if t[i] == 0.0:
    #        zero.append(x[i])
    #    else:
    #        one.append(x[i])

    #zero = np.array(zero)
    #one = np.array(one)

    #plt.scatter(zero[:,0], zero[:,1], label='t=zero')
    #plt.scatter(one[:,0], one[:,1], label='t=one')
    #plt.title("Valid y")
    #plt.xlabel("X_1")
    #plt.xticks(np.arange(minx-2, maxx + 2, (maxx-minx)/10))
    #plt.yticks(np.arange(miny-2, maxy+2, (maxy - miny)/10))
    #plt.ylabel("x_2")


    #valid_naive = LogisticRegression(theta_0=np.zeros([1, x.ndim]))
    #print("Starting valid fit with y at", dt.now())
    #valid_naive.fit(x, y)
    #print("Ending valid fit with y at", dt.now())

    #plotx = np.ones([1000, x.ndim])
    #plotx[:,0] = np.linspace(minx, miny, 1000)
    #plotx[:,1] = np.linspace(miny, maxy, 1000)
    #yset = valid_naive.predict(plotx)

    #plt.plot(plotx[:,1], yset, 'r-', label='Decision Boundry')
    #plt.legend()
    #plt.savefig("Q2ValidWY")
    #plt.figure()

    #t, x, y = loadData(train_path)
    #zero = []
    #one = []
    #minx = int(min(x[:,0]))
    #maxx = int(max(x[:,0]))
    #miny = int(min(x[:,1]))
    #maxy = int(max(x[:,1]))
    #for i in range(len(x)):
    #    if t[i] == 0.0:
    #        zero.append(x[i])
    #    else:
    #        one.append(x[i])

    #zero = np.array(zero)
    #one = np.array(one)

    #plt.scatter(zero[:,0], zero[:,1], label='t=zero')
    #plt.scatter(one[:,0], one[:,1], label='t=one')
    #plt.title("Train Y")
    #plt.xlabel("X_1")
    #plt.xticks(np.arange(minx-2, maxx + 2, (maxx-minx)/10))
    #plt.yticks(np.arange(miny-2, maxy+2, (maxy - miny)/10))
    #plt.ylabel("x_2")


    #train_naive = LogisticRegression(theta_0=np.zeros([1, x.ndim]))
    #print("Starting training data with y at", dt.now())
    #train_naive.fit(x, y)
    #print("Ending training with y at", dt.now())

    #plotx = np.ones([1000, x.ndim])
    #plotx[:,0] = np.linspace(minx, miny, 1000)
    #plotx[:,1] = np.linspace(miny, maxy, 1000)
    #yset = train_naive.predict(plotx)


    #plt.plot(plotx[:,1], yset, 'r-', label='Decision Boundry')
    #plt.legend()
    #plt.savefig("Q2ValidWY")

  # Part (f): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to output_path_adjusted

    plt.figure()
    t, x, y = loadData(valid_path)
    model = LogisticRegression(theta_0=np.zeros([1, x.ndim]))
    model.fit(x, y)

    plt.title("Alpha correction")
    plt.xlabel("X_1")
    plt.ylabel("X_2")

    validation = [x[i] for i in range(len(y)) if y[i] == 1]
    alpha = 1/len(validation)
    alpha *= np.sum(model.predict(validation))

    t, x, y = loadData(test_path)
    y = model.predict(x)
    y *= alpha


    plt.scatter([x[i][0] for i in range(len(x)) if t[i] == 0],
                [x[i][1] for i in range(len(x)) if t[i] == 0],
                label='t=0')
    plt.scatter([x[i][0] for i in range(len(x)) if t[i] == 1],
                [x[i][1] for i in range(len(x)) if t[i] == 1],
                label='t=1')

    plt.plot(x[:, 0], y, 'r-')
    np.savetxt(output_path_adjusted, y)
    plt.legend()
    plt.savefig("Q2Adjusted")
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
    return (np.array(ts), np.array(xs), np.array(ys))



if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
