from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import defaultdict


def get_data():
    categories = ['talk.religion.misc', 'soc.religion.christian', 'sci.space',
                  'comp.graphics']
    train = fetch_20newsgroups(subset='train', categories=categories)
    test = fetch_20newsgroups(subset='test', categories=categories)
    return train, test


def train_model():

    vec = CountVectorizer()
    vec.fit(train.data)
    train_data_trans = vec.transform(train.data)
    test_data_trans = vec.transform(test.data)
    train_target = train.target

    nbc = MultinomialNB()
    nbc.fit(train_data_trans, train_target)

    return nbc.predict(test_data_trans)


def conf_matrix(test_pred):
    return confusion_matrix(test.target, test_pred)


def conf_matrix_manual(test_pred):
    matrix = defaultdict(lambda: [0, 0, 0, 0])
    for label_act, label_pred in zip(test.target, test_pred):
        matrix[label_act][label_pred] += 1
    return matrix


if __name__ == '__main__':
    train, test = get_data()
    test_pred = train_model()

    print(conf_matrix_manual(test_pred))

