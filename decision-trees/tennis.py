import csv
from collections import namedtuple, defaultdict
import decision_trees


def get_tennis_data():
    tennis_data = []
    with open('tennis.csv') as f:
        f_csv = csv.reader(f)
        headings = next(f_csv)
        Tennis = namedtuple('Tennis', headings)
        for r in f_csv:
            tennis_data.append(Tennis(*r))
            # print('{:3s} {:8s} {:4s} {:6s} {:6s} {}'.format(*r))
    return tennis_data

if __name__ == '__main__':
    tennis_data = get_tennis_data()

    # starting entropy
    tennis_play_labels = [tennis_row.play_tennis for tennis_row in tennis_data]
    probabilities = decision_trees.class_probabilities(tennis_play_labels)
    entropy_init = decision_trees.entropy(probabilities)
    print(entropy_init)

    # humidity entropy
    entropy_hum = {}
    total_len = 0
    for hlevel in ['high', 'normal']:
        labels = ([tennis_row.play_tennis for tennis_row in tennis_data
                            if tennis_row.humidity == hlevel])
        total_len += len(labels)
        entropy_hum[hlevel] = (decision_trees.entropy(
            decision_trees.class_probabilities(labels)), len(labels))
    print(entropy_hum, total_len)

    gain_hum = entropy_init
    for hlevel in entropy_hum:
        e, l = entropy_hum[hlevel]
        gain_hum -= e * l / total_len
    print(gain_hum)

