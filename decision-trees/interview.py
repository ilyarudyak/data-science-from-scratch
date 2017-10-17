import csv
import decision_trees
from collections import namedtuple, defaultdict

headers = ['level', 'lang', 'tweets', 'phd', 'result']
attributes = headers[:-1]
interviews = [
    ({'level': 'Senior', 'lang': 'Java', 'tweets': 'no', 'phd': 'no'}, False),
    ({'level': 'Senior', 'lang': 'Java', 'tweets': 'no', 'phd': 'yes'}, False),
    ({'level': 'Mid', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, True),
    ({'level': 'Junior', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, True),
    ({'level': 'Junior', 'lang': 'R', 'tweets': 'yes', 'phd': 'no'}, True),
    ({'level': 'Junior', 'lang': 'R', 'tweets': 'yes', 'phd': 'yes'}, False),
    ({'level': 'Mid', 'lang': 'R', 'tweets': 'yes', 'phd': 'yes'}, True),
    ({'level': 'Senior', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, False),
    ({'level': 'Senior', 'lang': 'R', 'tweets': 'yes', 'phd': 'no'}, True),
    ({'level': 'Junior', 'lang': 'Python', 'tweets': 'yes', 'phd': 'no'}, True),
    ({'level': 'Senior', 'lang': 'Python', 'tweets': 'yes', 'phd': 'yes'}, True),
    ({'level': 'Mid', 'lang': 'Python', 'tweets': 'no', 'phd': 'yes'}, True),
    ({'level': 'Mid', 'lang': 'Java', 'tweets': 'yes', 'phd': 'no'}, True),
    ({'level': 'Junior', 'lang': 'Python', 'tweets': 'no', 'phd': 'yes'}, False)
]


def write_to_file():
    with open('interview.csv', 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows([get_row(interview) for interview in interviews])


def get_row(interview):
    d, result = interview
    return [d[header] for header in headers[:-1]] + [result]


def get_interviews():
    interviews = []
    with open('interview.csv') as f:
        f_csv = csv.reader(f)
        next(f_csv)
        attributes, label = headers[:-1], headers[-1]
        Row = namedtuple('Row', headers)
        for r in f_csv:
            row = Row(*r)
            d = {attribute: getattr(row, attribute) for attribute in attributes}
            interviews.append((d, getattr(row, label)))
    return interviews


def partition_by(interviews, attribute):
    """ for attribute == 'level':
        returns {'Junior': [(), (), ... ],
                'Mid': [...],
                "Senior: [...]}"""
    partitions_dic = defaultdict(list)
    for interview in interviews:
        interview_data, _ = interview
        attribute_value = interview_data[attribute]
        partitions_dic[attribute_value].append(interview)
    return dict(partitions_dic)


def partition_entropy_by(interviews, attribute):
    """computes the entropy corresponding to the given partition"""
    partitions = partition_by(interviews, attribute).values()
    return partition_entropy(partitions)


def partition_entropy(partitions):
    entropy = 0
    total_len = sum([len(labeled_data) for labeled_data in partitions])
    for labeled_data in partitions:
        weight = len(labeled_data) / total_len
        entropy += decision_trees.data_entropy(labeled_data) * weight
    return entropy


if __name__ == '__main__':
    for level in ['Senior', 'Junior']:
        level_interviews = partition_by(interviews, 'level')[level]
        print('{}:'.format(level))
        for attribute in attributes:
            if not attribute == 'level':
                entropy = partition_entropy_by(level_interviews, attribute)
                print('{:6s}: {:.4f}'.format(attribute, entropy))
        print()
