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

############# prior and likelihood ################
def get_prior():
    p_spam = (sum(labels) + 1) / len(labels)
    return {0: 1 - p_spam, 1: p_spam}

def get_likelihood():
    pass

############# posterior ###########################
def get_posterior():
    pass

############# testing #############################


if __name__ == '__main__':
    root_path = os.path.expanduser('~/data/spam_packt/enron1/')

    emails, labels = get_data()
    # X (sparse matrix, array has the shape (5171, 500)) and
    # cv.vocabulary_ (dictionary word:index of len 500)
    term_docs, feature_mapping = preprocess_emails()
    print(term_docs.toarray().shape)
    prior = get_prior()
    print(prior)

