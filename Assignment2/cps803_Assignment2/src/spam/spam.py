import collections

import numpy as np

import util
import svm

import pdb

class Model:
    def __init__(self, theta_0=None, px1y1=None, px1y0=None):
        self.theta = theta_0
        self.px1y1 = px1y1
        self.px1y0 = px1y0

    def fit(self, matrix, labels):

        count, dim = matrix.shape

        y = float(labels.sum()) # sum{y=1}
        ny = float(len(labels) - y) # sum{y=0}

        l_count = len(labels)
        py = y / l_count # probability of y=1
        pny = ny / l_count # probability of y=0
        if None == self.px1y0:
            self.px1y0 = np.zeros((dim,), dtype="float")
        if None == self.px1y1:
            self.px1y1 = np.zeros((dim,), dtype="float")

        for j in range(dim):
            for i in range(count):
                if labels[i] == 0:
                    if self.px1y0[j] == 0:
                        self.px1y0[j] = 2.0/(ny +2.0)
                    else:
                        self.px1y0[j] += 1.0/ny
                else:
                    if self.px1y1[j] == 0:
                        self.px1y1[j] = 2.0/(y+2.0)
                    else:
                        self.px1y1[j] += 1.0/y


        py1x1 = []
        for i in range(dim):
                py1x1.append(
                    self.px1y1[i] * py /
                    (self.px1y1[i] * py + self.px1y0[i]*pny)
                )

        self.theta = np.array(py1x1)

    def predict(self, matrix):
        res = []
        for x in matrix:
            try:
                res.append(1/(1+np.exp(-self.theta.dot(x))))
            except:
                pdb.set_trace()
        return np.array(res)


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    string = message.split(" ")
    output = []
    for x in string:
        output.append(x.lower())
    return output
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    counts = {}
    for message in messages:
        for word in get_words(message):
            if word not in counts.keys():
                counts[word] = 0
            counts[word] += 1
    return counts

    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message.
    Each row in the resulting array should correspond to each message
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    out = []
    keys = word_dictionary.keys()
    for message in messages:
        wordset = np.zeros(len(keys))
        i = 0
        words = get_words(message)
        for word in keys:
            if word in words:
                wordset[i] = word_dictionary[word]
            i += 1
        out.append(np.array(wordset))
    return np.array(out)


    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """
    # *** START CODE HERE ***
    model = Model()
    model.fit(matrix, labels)
    return model
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    return model.predict(matrix)
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    keys = list(dictionary.keys())
    res = []
    for x in range(len(keys)):
        res.append(((np.log(model.px1y1[x]/model.px1y0[x])), x))
    out = [(prob, keys[x]) for prob, x in sorted(res,
                                                 reverse=True)[0:4]]
    return out


    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
