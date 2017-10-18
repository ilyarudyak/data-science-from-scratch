from collections import Counter, defaultdict
from machine_learning import split_data
import math, random, re, glob
import numpy as np


def tokenize_message(message, token_pattern=r'(?u)\b\w\w+\b'):
    all_words = tokenize_message_to_list(message, token_pattern)
    return set(all_words)  # remove duplicates


def tokenize_message_to_list(message, token_pattern=r'(?u)\b\w\w+\b'):
    message = message.lower()  # convert to lowercase
    all_words = re.findall(token_pattern, message)  # extract the words
    return all_words


def count_words(training_set):
    """
    :parameter training_set is set of pairs (message, is_spam);
    :returns {'viagra': [100, 1], 'data': [1, 100], ... }
    """
    counts = defaultdict(lambda: [0, 0])
    for message, is_spam in training_set:
        for word in tokenize_message(message):

            counts[word][0 if is_spam else 1] += 1
    return counts


def build_vocabulary(training_set):
    vocabulary = count_words(training_set)
    sorted_words = sorted(vocabulary.keys())
    for index, word in enumerate(sorted_words):
        vocabulary[word] = index
    return vocabulary


def word_probabilities(counts, total_spam_msgs, total_ham_msgs, k=0.5):
    """
    turn the word_counts into a list of triplets
    w, p(w | spam) and p(w | ~spam);
    we use Laplace smoothing with parameter 'k';
    [('viagra', .7, .1), ...]
    """
    return [(w,
             (count_in_spam + k) / (total_spam_msgs + 2 * k),
             (count_in_ham + k) / (total_ham_msgs + 2 * k))
            for w, (count_in_spam, count_in_ham) in counts.items()]


def spam_probability(word_probs, message):
    message_words = tokenize_message(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0

    for word, prob_if_spam, prob_if_not_spam in word_probs:
        # for each word in the message,
        # add the log probability of seeing it
        if word in message_words:
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_not_spam += math.log(prob_if_not_spam)

        # for each word that's not in the message
        # add the log probability of _not_ seeing it
        else:
            log_prob_if_spam += math.log(1.0 - prob_if_spam)
            log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)

    prob_if_spam = math.exp(log_prob_if_spam)
    prob_if_not_spam = math.exp(log_prob_if_not_spam)
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)


class NaiveBayesClassifier:
    def __init__(self, k=1.0):
        self.k = k
        self.word_probs = []  # [('handheld', 0.004054054054054054, 0.0002294630564479119) ...
        self.word_counts = {}  # {'handheld': [1, 0], 'you': [31, 20], 'organizer': [1, 0] ...

    def train(self, training_set):
        # count spam and non-spam messages
        num_spams = len([is_spam
                         for message, is_spam in training_set
                         if is_spam])
        num_non_spams = len(training_set) - num_spams

        # run training data through our "pipeline"
        self.word_counts = dict(count_words(training_set))
        self.word_probs = word_probabilities(self.word_counts,
                                             num_spams,
                                             num_non_spams,
                                             self.k)

    def classify(self, message):
        return spam_probability(self.word_probs, message)


def get_subject_data(path):
    data = []

    # regex for stripping out the leading "Subject:" and any spaces after it
    subject_regex = re.compile(r"^Subject:\s+")

    # glob.glob returns every filename that matches the wildcarded path
    for fn in glob.glob(path):
        is_spam = "ham" not in fn

        with open(fn, 'r', encoding='ISO-8859-1') as file:
            for line in file:
                if line.startswith("Subject:"):
                    subject = subject_regex.sub("", line).strip()
                    data.append((subject, is_spam))

    return data


def get_train_test_data(path):
    data = get_subject_data(path)
    random.seed(0)  # just so you get the same answers as me
    train_data, test_data = split_data(data, 0.75)
    return train_data, test_data


def p_spam_given_word(word_prob):
    word, prob_if_spam, prob_if_not_spam = word_prob
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)


def train_and_test_model(path):

    data = get_subject_data(path)
    random.seed(0)  # just so you get the same answers as me
    train_data, test_data = split_data(data, 0.75)

    classifier = NaiveBayesClassifier()
    classifier.train(train_data)

    classified = [(subject, is_spam, classifier.classify(subject))
                  for subject, is_spam in test_data]

    counts = Counter([(is_spam, spam_probability > 0.5)  # (actual, predicted)
                     for _, is_spam, spam_probability in classified])

    print(counts, '\n')

    classified.sort(key=lambda row: row[2])
    spammiest_hams = list(filter(lambda row: not row[1], classified))[-10:]
    hammiest_spams = list(filter(lambda row: row[1], classified))[:10]

    print("spammiest_hams: ", list(map(lambda row: row[0], spammiest_hams)), '\n')
    print("hammiest_spams: ", list(map(lambda row: row[0], hammiest_spams)), '\n')

    words = sorted(classifier.word_probs, key=p_spam_given_word)

    spammiest_words = words[-10:]
    hammiest_words = words[:10]

    print("spammiest_words", list(map(lambda row: row[0], spammiest_words)))
    print("hammiest_words", list(map(lambda row: row[0], hammiest_words)))


def train_simple_set():
    # simple training set
    spam_message = 'rolex viagra free money'
    spam_message2 = 'rolex'
    ham_message = 'data'
    ham_message2 = 'data data'
    training_set = {(spam_message, 1), (spam_message2, 1), (ham_message, 0), (ham_message2, 0)}
    test_message = 'rolex money'

    nbc = NaiveBayesClassifier(k=1.0)
    nbc.train(training_set)
    print(nbc.classify(test_message))


def train_and_test_model2(path):
    data = get_subject_data(path)
    random.seed(0)  # just so you get the same answers as me
    train_data, test_data = split_data(data, 0.75)

    nbc = NaiveBayesClassifier()
    nbc.train(train_data)

    classified = [(subject, is_spam, nbc.classify(subject))
                  for subject, is_spam in test_data]

    counts = Counter((is_spam, spam_probability > 0.5)  # (actual, predicted)
                     for _, is_spam, spam_probability in classified)

    print(counts)

    return np.array([spam_probability > 0.5 for _, _, spam_probability in classified]), \
           np.array([prob for _, _, prob in classified])


if __name__ == "__main__":
    # data is here: http://spamassassin.apache.org/old/publiccorpus/
    path = r'/Users/ilyarudyak/Downloads/*/*'

    train_and_test_model2(path)



