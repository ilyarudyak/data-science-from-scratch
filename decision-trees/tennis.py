import csv
from collections import namedtuple, defaultdict
import decision_trees


def get_tennis_data():
    tennis_data = []
    with open('tennis.csv') as f:
        f_csv = csv.reader(f)
        headings = next(f_csv)
        Tennis = namedtuple('Tennis', headings)
        for row in f_csv:
            tennis_data.append(Tennis(*row))
            # print('{:3s} {:8s} {:4s} {:6s} {:6s} {:3s}'.format(*r))
    return tennis_data


def get_first_level_outcomes(tennis_data):
    return {heading: count_outcomes(tennis_data, heading) for heading in tennis_data[0]._fields[1:-1]}


def count_outcomes(tennis_data, heading):
    outcomes = defaultdict(lambda: [0, 0, 0])
    for tennis_row in tennis_data:
        outcomes[getattr(tennis_row, heading)][0 if tennis_row.play_tennis == 'yes' else 1] += 1

    for outcome in outcomes:
        total = outcomes[outcome][0] + outcomes[outcome][1]
        outcomes[outcome][0] /= total
        outcomes[outcome][1] /= total
        outcomes[outcome][2] = total

    return dict(outcomes)


def get_first_level_gains(tennis_data):
    entropy0, total_labels = get_zero_level_entropy(tennis_data)
    outcomes = get_first_level_outcomes(tennis_data)
    gains = {}
    for heading in outcomes:
        # print('{}:'.format(heading))
        conditions = outcomes[heading]
        cond_entropy = 0
        for condition in conditions:
            entropy = decision_trees.entropy(conditions[condition][0:2])
            # print('{}: e={:.3f}'.format(condition, entropy))
            cond_entropy += entropy * conditions[condition][2] / total_labels
        # print('cond_entropy={:.3f} gain={:.3f}\n'.format(cond_entropy, entropy0 - cond_entropy))
        gains[heading] = entropy0 - cond_entropy
    return gains


def get_zero_level_entropy(tennis_data):
    labels = [getattr(tennis_row, 'play_tennis') for tennis_row in tennis_data]
    probabilities = decision_trees.class_probabilities(labels)
    return decision_trees.entropy(probabilities), len(labels)


if __name__ == '__main__':
    tennis_data = get_tennis_data()
    print(get_first_level_gains(tennis_data))
