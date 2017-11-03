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
def preprocess_emails():
    lemmatized_emails = [lemmatize_and_remove_names(email) for email in emails]
    term_docs, vocabulary = count_words(lemmatized_emails)
    with open('preprocessed_emails.p', 'wb') as f:
        pickle.dump(term_docs, f)

def get_preprocessed_emails():
    with open('preprocessed_emails.p', 'rb') as f:
        return pickle.load(f)

def lemmatize_and_remove_names(email):
    all_names = set(names.words())
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word.lower())
                     for word in email.split()
                     if word.isalpha() and
                     word not in all_names])

def count_words(lemmatized_emails):
    cv = CountVectorizer(stop_words='english', max_features=500)
    cv.fit(lemmatized_emails)

    return cv.transform(lemmatized_emails), cv.vocabulary_

############# prior and likelihood #################
def get_prior():
    p_spam = sum(labels) / len(labels)
    return {0: 1 - p_spam, 1: p_spam}

def get_likelihood(smoothing=1):
    return np.array([get_likelihood_from_label2(0, smoothing=smoothing),
                     get_likelihood_from_label2(1, smoothing=smoothing)])

def get_likelihood_from_label(label, smoothing=1):
    term_docs_array = term_docs.toarray()
    term_docs_array_label = term_docs_array[np.array(labels) == label, :]
    total_count = np.sum(term_docs_array_label) + term_docs_array.shape[1] * smoothing
    return (np.sum(term_docs_array_label, axis=0) + smoothing) / total_count

def get_likelihood_from_label2(label, smoothing=1):
    term_docs_label = term_docs[np.array(labels) == label, :]
    total_count = np.sum(term_docs_label) + 500 * smoothing
    likelihood = (np.sum(term_docs_label, axis=0) + smoothing) / total_count
    likelihood = np.asarray(likelihood)
    return likelihood[0]

# -------------------- from solution --------------

def get_label_index():
    from collections import defaultdict
    label_index = defaultdict(list)
    for index, label in enumerate(labels):
        label_index[label].append(index)
    return label_index

def get_likelihood2(smoothing=1):
    likelihood = {}
    for label, index in get_label_index().items():
        likelihood[label] = term_docs[index, :].sum(axis=0) + smoothing
        likelihood[label] = np.asarray(likelihood[label])[0]

        total_count = likelihood[label].sum()
        likelihood[label] = likelihood[label] / float(total_count)
    return likelihood

############# posterior ############################
def get_posterior():
    pass

############# testing ##############################


if __name__ == '__main__':
    root_path = os.path.expanduser('~/data/spam_packt/enron1/')

    emails, labels = get_data()
    term_docs = get_preprocessed_emails()

    # prior = get_prior()
    tic = time.time()
    likelihood = get_likelihood()
    toc = time.time()
    print(likelihood[0][:5], toc - tic)

    tic = time.time()
    likelihood2 = get_likelihood2()
    toc = time.time()
    print(likelihood2[0][:5], toc - tic)
