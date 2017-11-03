import numpy as np
import spam_packt.naive_bayes_manual

def get_likelihood():
    X_spam = X[labels == 1]
    num_features = X.shape[1]
    total_count = np.sum(X_spam) + num_features
    return (np.sum(X_spam, axis=0) + 1) / total_count


if __name__ == '__main__':
    X = np.array([[2, 0, 0, 0, 1],
                  [0, 0, 1, 1, 0],
                  [0, 0, 0, 1, 0],
                  [1, 0, 1, 1, 0],
                  [0, 0, 0, 0, 3],
                  [0, 1, 1, 0, 2],
                  [1, 2, 0, 0, 1]])

    labels = np.array([1, 1, 1, 1, 0, 0, 0])

    print('manual:', [4/14, 1/14, 3/14, 4/14, 2/14])
    print('local function:', get_likelihood())
