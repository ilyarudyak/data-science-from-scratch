from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

import glob
import os
import numpy as np


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
    return count_words(lemmatized_emails)

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

    feature_names = cv.get_feature_names()
    # print(len(feature_names), feature_names[481], feature_names[357])
    # print(feature_names[:10])
    # print(cv.vocabulary_)
    # print(cv.vocabulary)

    return cv.transform(lemmatized_emails), cv.vocabulary_

############# prior and likelihood #################
def get_prior():
    p_spam = sum(labels) / len(labels)
    return {0: 1 - p_spam, 1: p_spam}

def get_likelihood(smoothing=1):
    return np.array([get_likelihood_from_label(0, smoothing=smoothing),
                     get_likelihood_from_label(1, smoothing=smoothing)])

def get_likelihood_from_label(label, smoothing=1):
    term_docs_array = term_docs.toarray()
    term_docs_array_label = term_docs_array[np.array(labels) == label, :]
    total_count = np.sum(term_docs_array_label) + term_docs_array.shape[1] * smoothing
    return (np.sum(term_docs_array_label, axis=0) + smoothing) / total_count

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
    print(len(emails), len(labels))
    # X (sparse matrix, array has the shape (5171, 500)) and
    # cv.vocabulary_ (dictionary word:index of len 500)
    term_docs, feature_mapping = preprocess_emails()
    print(term_docs.toarray().shape)

    # prior = get_prior()
    likelihood = get_likelihood()
    print(len(likelihood[0]), likelihood[0][:5])

    likelihood2 = get_likelihood2()
    print(len(likelihood2[0]), likelihood2[0][:5])
