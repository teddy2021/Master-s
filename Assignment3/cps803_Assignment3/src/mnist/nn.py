import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import pdb
from datetime import datetime

def softmax(x):
    """
    Compute softmax function for a batch of input values.
    The first dimension of the input corresponds to the batch size. The second dimension
    corresponds to every class in the output. When implementing softmax, you should be careful
    to only sum over the second dimension.

    Important Note: You must be careful to avoid overflow for this function. Functions
    like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
    You will know that your function is overflow resistent when it can handle input like:
    np.array([[10000, 10010, 10]]) without issues.

    Args:
        x: A 2d numpy float array of shape batch_size x number_of_classes

    Returns:
        A 2d numpy float array containing the softmax results of shape batch_size x number_of_classes
    """
    # *** START CODE HERE ***

    ex = np.exp(x - np.max(x))
    return ex/np.sum(ex)

    # *** END CODE HERE ***

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the sigmoid results
    """
    # *** START CODE HERE ***
    return 1./(1. + np.exp(-x))
    # *** END CODE HERE ***

def get_initial_params(input_size, num_hidden, num_output):
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    There should be four parameters for this model:
    W1 is the weight matrix for the hidden layer of size input_size x num_hidden
    b1 is the bias vector for the hidden layer of size num_hidden
    W2 is the weight matrix for the output layers of size num_hidden x num_output
    b2 is the bias vector for the output layer of size num_output

    As specified in the PDF, weight matrices should be initialized with a random normal distribution
    centered on zero and with scale 1.
    Bias vectors should be initialized with zero.

    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes

    Returns:
        A dict mapping parameter names to numpy arrays
    """

    # *** START CODE HERE ***
    var = 1/math.sqrt(input_size)
    mean = 0
    W1 = np.random.normal(mean, var, (input_size, num_hidden))
    W2 = np.random.normal(mean, var, (num_hidden, num_output))
    b1 = np.zeros((1, num_hidden))
    b2 = np.zeros((1, num_output))

    di = {"lMats": W1, "lBias": b1, "oMats": W2, "oBias": b2}
    return di
    # *** END CODE HERE ***

def forward_prop(data, labels, params):
    """
    Implement the forward layer given the data, labels, and params.

    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after the sigmoid) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """
    # *** START CODE HERE ***
    layers = params['lMats']
    lbias = params['lBias']

    output = params['oMats']
    obias = params['oBias']

    loss_avg = 0
    activations = []
    results = []
    for i in range(len(data)):
        x = data[i].reshape((1, -1))
        # z = (m,300) = (m, 768) x (768, 300)
        z1 = x @ layers
        z1 += lbias
        activations.append(sigmoid(z1))
        # (m, 10) = (m, 300) (300, 10)
        z2 = activations[-1] @ output
        z2 += obias
        results.append(softmax(z2))


        # the below is intended for naive use with log, it prevents log of zero by
        # using a small constant
        # loss_avg = np.sum( (labels * np.log(y + (10**-10))))/len(data)
        loss_avg += -np.sum(labels[i] * np.where(labels[i] == 1,
                                        np.log(results[-1]), 0))/len(data)
    a = np.array(results)
    b = np.array(activations)
    res = np.array(results).reshape(a.shape[0], a.shape[2])
    yHat = np.array(activations).reshape(b.shape[0], b.shape[2])
    return (yHat, res, loss_avg )
    # *** END CODE HERE ***

def backward_prop(data, labels, params, forward_prop_func):
    """
    Implement the backward propegation gradient computation step for a neural network

    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.

        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    # Steps:
    # Get output values
    yHat, y, loss = forward_prop_func(data, labels, params)
    output = {}

    om = 0
    lm = 0
    ob = 0
    lb = 0
    for x in range(len(data)):
        curY = y[x].reshape((1,-1), order='F') # 1x10
        curYHat = yHat[x].reshape((1,-1), order='F') # 1xn
        # When the derivations are done, and then simplified for the output,
        # the result is yHat - y for the output error with respect to z.
        dout =  curY - labels[x].reshape(curY.shape) # 1x10
        om += curYHat.T @ dout # nx10
        ob += dout # 1xn
        # push the gradient backwards through the matrix to get to the output
        # matrix.
        dhid = dout @ params['oMats'].T # 1xm
        lm += data[x].reshape(1,-1, order='F').T @ dhid # mxn
        lb += dhid # 1xm
    output ['lMats'] = lm/len(data)
    output ['oMats'] = om/len(data)
    output ['lBias'] = lb/len(data)
    output ['oBias'] = ob/len(data)

    return output
    # *** END CODE HERE ***

def backward_prop_regularized(data, labels, params, forward_prop_func, reg):
    """
    Implement the backward propegation gradient computation step for a neural network

    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above
        reg: The regularization strength (lambda)

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.

        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    output = {}
    yHat, y, loss = forward_prop_func(data, labels, params)
    output = {}
    out = 0

    layers = 0
    lBias = 0
    oBias = 0

    om = 0
    lm = 0
    ob = 0
    lb = 0
    for x in range(len(data)):
        curY = y[x].reshape((1,-1), order='F') # 1x10
        curYHat = yHat[x].reshape((1,-1), order='F') # 1xn
        dout =  curY - labels[x].reshape(curY.shape) # 1x10
        om += curYHat.T @ dout + params['oMats'] # nx10
        ob += dout # 1xn

        dhid = dout @ params['oMats'].T # 1xm
        lm += data[x].reshape(1,-1, order='F').T @ dhid + params['lMats']# mxn
        lb += dhid # 1xm

    output['oMats'] = om/len(data)
    output['lMats'] = lm/len(data)
    output['oBias'] = ob/len(data)
    output['lBias'] = lb/len(data)
    return output

    # *** END CODE HERE ***

def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func):
    """
    Perform one epoch of gradient descent on the given training data using the provided learning rate.

    This code should update the parameters stored in params.
    It should not return anything

    Args:
        train_data: A numpy array containing the training data
        train_labels: A numpy array containing the training labels
        learning_rate: The learning rate
        batch_size: The amount of items to process in each batch
        params: A dict of parameter names to parameter values that should be updated.
        forward_prop_func: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API

    Returns: This function returns nothing.
    """

    # *** START CODE HERE ***
    change = []
    loss = np.zeros(
        (int(len(train_data)/batch_size),1)
    )
    for x in range(0, len(train_data), batch_size):
        end = x + batch_size
        # note, the below includes a /255 to normalize the data. This just
        # provides some buffer for the log and softmax functions to further
        # avoid overflow.
        res = backward_prop_func(train_data[x:end]/255,
            train_labels[x:end], params, forward_prop_func)
        change.append(res)



    lMatrix = 0
    oMatrix = 0
    lBias = 0
    oBias = 0
    iterations = len(train_data)/batch_size

    for x in range(len(change)):
        lMatrix += change[x]['lMats']
        oMatrix += change[x]['oMats']
        lBias += change[x]['lBias']
        oBias += change[x]['oBias']

    params['lMats'] -= lMatrix * learning_rate/iterations
    params['oMats'] -= oMatrix * learning_rate/iterations
    params['lBias'] -= lBias * learning_rate/iterations
    params['oBias'] -= oBias * learning_rate/iterations
    # *** END CODE HERE ***

    # This function does not return anything
    return

def nn_train(
    train_data, train_labels, dev_data, dev_labels,
    get_initial_params_func, forward_prop_func, backward_prop_func,
    num_hidden=300, learning_rate=5, num_epochs=30, batch_size=1000, num_out=10):

    (nexp, dim) = train_data.shape

    params = get_initial_params_func(dim, num_hidden, num_out)

    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []
    for epoch in range(num_epochs):
        print(epoch, ': ', datetime.now(), sep='')
        gradient_descent_epoch(train_data, train_labels,
        learning_rate, batch_size, params, forward_prop_func, backward_prop_func)

        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output,train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))

    return params, cost_train, cost_dev, accuracy_train, accuracy_dev

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) ==
        np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def read_data(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def dummy_params(dim, n_hid, n_out):
    print(dim, n_hid, n_out)
    dic = {}
    dic['lMats'] = np.ones((dim, n_hid))
    dic['oMats'] = np.ones((n_hid,n_out))
    dic['lBias'] = np.zeros((n_hid, 1))
    dic['oBias'] = np.zeros((n_out, 1))
    return dic

def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs, plot=True):
#    data = np.array([[0,0],
#                     [1,0],
#                     [1,1],
#                     [0,1]])
#    labels = np.array([[0,1],
#                      [1,0],
#                      [1,0],
#                      [1,0]])
#
#
#    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
#        data, labels,
#        data, labels,
#        dummy_params, forward_prop, backward_prop_func,
#        num_hidden=3, learning_rate=5, num_epochs=15, batch_size=2, num_out=2
#    )
#    print(params)
#
#
#    accuracy= nn_test(data, labels, params)
#    print('Got', accuracy, 'accuracy for dummy run')
#    exit()
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'],
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=300, learning_rate=5, num_epochs=num_epochs, batch_size=1000
    )

    t = np.arange(num_epochs)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(t, cost_train,'r', label='train')
        ax1.plot(t, cost_dev, 'b', label='dev')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        if name == 'baseline':
            ax1.set_title('Without Regularization')
        else:
            ax1.set_title('With Regularization')
        ax1.legend()

        ax2.plot(t, accuracy_train,'r', label='train')
        ax2.plot(t, accuracy_dev, 'b', label='dev')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig('./' + name + '.pdf')

    accuracy = nn_test(all_data['test'], all_labels['test'], params)
    print('For model %s, got accuracy: %f' % (name, accuracy))

    return accuracy

def main(plot=True):
    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=30)

    args = parser.parse_args()

    np.random.seed(100)
    train_data, train_labels = read_data('./images_train.csv', './labels_train.csv')
    train_labels = one_hot_labels(train_labels)
    p = np.random.permutation(60000)
    train_data = train_data[p,:]
    train_labels = train_labels[p,:]

    dev_data = train_data[0:10000,:]
    dev_labels = train_labels[0:10000,:]
    train_data = train_data[10000:,:]
    train_labels = train_labels[10000:,:]

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    test_data, test_labels = read_data('./images_test.csv', './labels_test.csv')
    test_labels = one_hot_labels(test_labels)
    test_data = (test_data - mean) / std

    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }
    baseline_acc = run_train_test('baseline', all_data, all_labels, backward_prop, args.num_epochs, plot)
    reg_acc = run_train_test('regularized', all_data, all_labels,
        lambda a, b, c, d: backward_prop_regularized(a, b, c, d, reg=0.0001),
        args.num_epochs, plot)

    return baseline_acc, reg_acc

if __name__ == '__main__':
    main()
