from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

import glob
import os
import numpy as np
import time
import pickle


def get_samples():
    paths = [os.path.join(root_path, 'ham/0007.1999-12-14.farmer.ham.txt'),
             os.path.join(root_path + 'spam/0058.2003-12-21.GP.spam.txt')]
    samples = []
    for path in paths:
        with open(path, 'r') as f:
            samples.append(f.read())
    return samples

def get_data():
    emails, labels = [], []  # spam is 1; ham is 0
    for category in ['spam', 'ham']:
        for filename in glob.glob(os.path.join(root_path, category + '/*')):
            with open(filename, 'r', encoding='ISO-8859-1') as f:
                emails.append(f.read())
                labels.append(1 if category == 'spam' else 0)
    return emails, labels

############# preprocessing #######################
def clean_emails(emails):
    return [lemmatize_and_remove_names(email) for email in emails]

def lemmatize_and_remove_names(email):
    all_names = set(names.words())
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word.lower())
                     for word in email.split()
                     if word.isalpha() and
                     word not in all_names])

def count_words(lemmatized_emails):
    cv.fit(lemmatized_emails)
    return cv.transform(lemmatized_emails)

def preprocess_test_emails(test_emails):
    return cv.transform(clean_emails(test_emails))

############# prior and likelihood #################
def get_prior(labels):
    p_spam = sum(labels) / len(labels)
    return {0: 1 - p_spam, 1: p_spam}

def get_likelihood(X, labels, smoothing=1):
    return {0: get_likelihood_from_label2(X, labels, 0, smoothing=smoothing),
            1: get_likelihood_from_label2(X, labels, 1, smoothing=smoothing)}

def get_likelihood_from_label(X, labels, label, smoothing=1):
    term_docs_array = X.toarray()
    term_docs_array_label = term_docs_array[np.array(labels) == label, :]
    total_count = np.sum(term_docs_array_label) + term_docs_array.shape[1] * smoothing
    return (np.sum(term_docs_array_label, axis=0) + smoothing) / total_count

def get_likelihood_from_label2(X, labels, label, smoothing=1):
    term_docs_label = X[np.array(labels) == label, :]
    total_count = np.sum(term_docs_label) + 500 * smoothing
    likelihood = (np.sum(term_docs_label, axis=0) + smoothing) / total_count
    likelihood = np.asarray(likelihood)
    return likelihood[0]

# -------------------- from solution --------------

def get_label_index(labels):
    from collections import defaultdict
    label_index = defaultdict(list)
    for index, label in enumerate(labels):
        label_index[label].append(index)
    return label_index

def get_likelihood_solution(X, labels, smoothing=1):
    likelihood = {}
    for label, index in get_label_index(labels).items():
        likelihood[label] = X[index, :].sum(axis=0) + smoothing
        likelihood[label] = np.asarray(likelihood[label])[0]

        total_count = likelihood[label].sum()
        likelihood[label] = likelihood[label] / float(total_count)
    return likelihood

############# posterior ############################
def get_posterior(X, labels, prior, likelihood):
    """

    :param X:
    :param labels:
    :param prior: dictionary {0: 1 - p(spam), 1: p(spam)}
    :param likelihood:  dictionary;
                        likelihood[0] - list of p(word|~spam);
                        likelihood[1] - list of p(word|spam);
    :return: list of dictionaries for each training example;
             each dictionary is of the form {0: p(~spam), 1: p(spam)}
    """
    num_docs = X.shape[0]
    posteriors = []
    for i in range(num_docs):
        posterior = {key: np.log(prior_label) for key, prior_label in prior.items()}
        return

# -------------------- from solution --------------

def get_posterior_sol(term_document_matrix, prior, likelihood):
    """ Compute posterior of testing samples, based on prior and likelihood
    Args:
        term_document_matrix (sparse matrix)
        prior (dictionary, with class label as key, corresponding prior as the value)
        likelihood (dictionary, with class label as key, corresponding conditional probability vector as value)
    Returns:
        dictionary, with class label as key, corresponding posterior as value
    """
    num_docs = term_document_matrix.shape[0]
    posteriors = []
    for i in range(num_docs):
        # posterior is proportional to prior * likelihood
        # = exp(log(prior * likelihood))
        # = exp(log(prior) + log(likelihood))
        posterior = {key: np.log(prior_label) for key, prior_label in prior.items()}
        for label, likelihood_label in likelihood.items():
            term_document_vector = term_document_matrix.getrow(i)
            counts = term_document_vector.data
            indices = term_document_vector.indices
            for count, index in zip(counts, indices):
                posterior[label] += np.log(likelihood_label[index]) * count
        # exp(-1000):exp(-999) will cause zero division error,
        # however it equates to exp(0):exp(1)
        min_log_posterior = min(posterior.values())
        for label in posterior:
            try:
                posterior[label] = np.exp(posterior[label] - min_log_posterior)
            except:
                # if one's log value is excessively large, assign it infinity
                posterior[label] = float('inf')
        # normalize so that all sums up to 1
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors


############# testing ##############################

def get_test_emails():
    test_emails = []
    for filename in glob.glob('spam_packt/email_test*.txt'):
        with open(filename, 'r') as f:
            test_emails.append(f.read())
    return test_emails


if __name__ == '__main__':
    root_path = os.path.expanduser('~/data/spam_packt/enron1/')
    cv = CountVectorizer(stop_words='english', max_features=500)

    enron1_emails, enron1_labels = get_data()
    enron1_term_docs = count_words(clean_emails(enron1_emails))

    prior = get_prior(enron1_labels)
    likelihood = get_likelihood(enron1_term_docs, enron1_labels)
    print(likelihood[0][:5])

    test_emails = get_test_emails()
    print(test_emails)
    term_docs_test = preprocess_test_emails(test_emails)
    posterior = get_posterior_sol(term_docs_test, prior, likelihood)
    print(posterior)


