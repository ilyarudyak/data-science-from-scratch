import naive_bayes
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


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


def split_data(path):
    train_data, test_data = naive_bayes.get_train_test_data(path)
    train_data, test_data = np.array(train_data), np.array(test_data)
    train_data_X, train_data_y, test_data_X, test_data_y = \
        train_data[:, 0], train_data[:, 1], test_data[:, 0], test_data[:, 1]
    return train_data_X, train_data_y, test_data_X, test_data_y


def diff_in_vocabulary(path):
    train_data, test_data = naive_bayes.get_train_test_data(path)
    train_data_X, train_data_y, test_data_X, test_data_y = split_data(path)

    vocabulary_my = naive_bayes.count_words_like_CountVectorizer(train_data)
    vocabulary_sklearn = build_CountVectorizer(train_data_X).vocabulary_
    print(len(vocabulary_my), len(vocabulary_sklearn))

    # for word in vocabulary_my:
    #     if vocabulary_my[word] != vocabulary_sklearn[word]:
    #         print(word, vocabulary_my[word], vocabulary_sklearn[word])


if __name__ == '__main__':
    path = r'/Users/ilyarudyak/Downloads/*/*'

    diff_in_vocabulary(path)
