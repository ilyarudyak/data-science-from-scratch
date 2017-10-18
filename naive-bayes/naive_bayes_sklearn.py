import naive_bayes
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict, Counter


def split_data(path):
    train_data, test_data = naive_bayes.get_train_test_data(path)
    train_data, test_data = np.array(train_data), np.array(test_data)
    train_data_X, train_data_y, test_data_X, test_data_y = \
        train_data[:, 0], train_data[:, 1], test_data[:, 0], test_data[:, 1]
    return train_data_X, train_data_y, test_data_X, test_data_y


path = r'/Users/ilyarudyak/Downloads/*/*'
train_data, test_data = naive_bayes.get_train_test_data(path)
train_data_X, train_data_y, test_data_X, test_data_y = split_data(path)


def test_CountVectorizer():
    corpus = np.array(['This is the first document.',
                       'This is the second second document.',
                       'And the third one.',
                       'Is this the first document?'])

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)

    print(vectorizer.get_feature_names())
    print(vectorizer.vocabulary_)
    print(X.toarray())
    print(vectorizer.transform(['this is the fifth document']).toarray())


def build_CountVectorizer(corpus):
    vectorizer = CountVectorizer()
    vectorizer.fit(corpus)
    return vectorizer


def diff_in_vocabulary(path):
    vocabulary_my = naive_bayes.build_vocabulary(train_data)
    vocabulary_sklearn = build_CountVectorizer(train_data_X).vocabulary_
    print(len(vocabulary_my), len(vocabulary_sklearn))

    for word in vocabulary_my:
        if vocabulary_my[word] != vocabulary_sklearn[word]:
            print(word, vocabulary_my[word], vocabulary_sklearn[word])


def transform(message, vocabulary):
    x = np.zeros(len(vocabulary))
    for word in naive_bayes.tokenize_message_to_list(message[0]):
        if word in vocabulary:
            x[vocabulary[word]] += 1
    return x.reshape(1, len(vocabulary))


def diff_in_transform(message):
    vocabulary_my = naive_bayes.build_vocabulary(train_data)
    x_my = transform(message, vocabulary_my)
    x_sk = build_CountVectorizer(train_data_X).transform(message)
    x_sk = x_sk.toarray()
    print(x_my.shape, x_sk.shape)
    print(np.sum(x_my), np.sum(x_sk))
    print(np.sum(x_my == x_sk) == len(vocabulary_my))


if __name__ == '__main__':

    message = ['lowest life insurance rates life']
    diff_in_transform(message)

