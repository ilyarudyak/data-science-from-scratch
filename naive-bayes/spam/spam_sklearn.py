import glob
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix


def get_email_data(path):
    """
    :param path: path to email directories
    :return: data: list of tuples (email text, is_spam)
    """
    data, labels = [], []
    NEWLINE = '\n'

    # glob.glob returns every filename that matches the wildcarded path
    for fn in glob.glob(path):
        is_spam = "ham" not in fn
        is_text, text = False, []
        with open(fn, 'r', encoding='ISO-8859-1') as file:
            for line in file:
                if is_text:
                    text.append(line.strip())
                if line.startswith(NEWLINE):
                    is_text = True
        data.append(" ".join(text))
        labels.append(is_spam)

    return np.array(data), np.array(labels)


def split_data():
    return cross_validation.train_test_split(email_data, labels, test_size=.1, random_state=42)


def train_model():
    vec = CountVectorizer()
    vec.fit(data_train)
    data_train_trans = vec.transform(data_train)
    data_test_trans = vec.transform(data_test)

    nbc = MultinomialNB()
    nbc.fit(data_train_trans, labels_train)
    return nbc.predict(data_test_trans)


def conf_matrix():
    return confusion_matrix(labels_test, test_pred)


if __name__ == '__main__':
    path = r'/Users/ilyarudyak/Downloads/*/*'
    email_data, labels = get_email_data(path)
    data_train, data_test, labels_train, labels_test = split_data()

    test_pred = train_model()
    print(conf_matrix())
