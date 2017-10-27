import glob
import os
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer


def get_samples():
    paths = [os.path.join(root_path, 'ham/0007.1999-12-14.farmer.ham.txt'),
             os.path.join(root_path + 'spam/0058.2003-12-21.GP.spam.txt')]
    samples = []
    for path in paths:
        with open(path, 'r') as f:
            samples.append(f.read())
    return samples


def get_data():
    emails, labels = [], []
    for category in ['spam', 'ham']:
        for filename in glob.glob(os.path.join(root_path, category + '/*')):
            with open(filename, 'r', encoding='ISO-8859-1') as f:
                emails.append(f.read())
                labels.append(1 if category == 'spam' else 0)
    return emails, labels


def preprocess_data():
    return [lemmatize_and_remove_names(email) for email in emails]


def lemmatize_and_remove_names(email):
    all_names = set(names.words())
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word.lower())
                     for word in email.split()
                     if word.isalpha() and
                     word not in all_names])


if __name__ == '__main__':
    root_path = os.path.expanduser('~/data/spam_packt/enron1/')
    emails, labels = get_data()
    print(len(emails), len(labels))
    print(emails[:3])
