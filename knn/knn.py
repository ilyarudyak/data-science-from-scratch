from collections import Counter, namedtuple
from linear_algebra import distance
from plot_state_borders import plot_state_borders
from stats import mean
import math
import random
import matplotlib.pyplot as plt
import csv


def raw_majority_vote(labels):
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner


def majority_vote(labels):
    """assumes that labels are ordered from nearest to farthest"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])

    if num_winners == 1:
        return winner  # unique winner, so return it
    else:
        return majority_vote(labels[:-1])  # try again without the farthest


def knn_classify(k, labeled_points, new_point):
    """
    each labeled point should be a pair (point, label)
    in our case:
     - labeled_points = [([longitude, latitude], label), ... ]
     - by_distance - sorted labeled_points by distance to new_point
       (in ascending order, so closest points are in the beginning)
     - labeled_point[0] = [longitude, latitude]; new_point = [longitude, latitude]
     - distance - standard euclidean distance (squared distance of coordinates)
    """

    # order the labeled points from nearest to farthest
    by_distance = sorted(labeled_points,
                         key=lambda labeled_point: distance(labeled_point[0], new_point))

    # find the labels for the k closest (as mentioned, closest points are in the
    # beginning of the list); by_distance = [([longitude, latitude], language), ... ]
    # so we unpack tuple ([longitude, latitude], language) into _, label
    # we can also write [point[1] for point in by_distance[:k]]
    k_nearest_labels = [label for _, label in by_distance[:k]]

    # and let them vote
    return majority_vote(k_nearest_labels)


def get_cities(filename):
    with open(filename) as f:
        f_csv = csv.reader(f)
        headings = next(f_csv)
        City = namedtuple('City', headings)
        return [City(*r) for r in f_csv]


def plot_cities():
    # key is language, value is pair (longitudes, latitudes)
    plots = {"Java": ([], []), "Python": ([], []), "R": ([], [])}

    # we want each language to have a different marker and color
    markers = {"Java": "o", "Python": "s", "R": "^"}
    colors = {"Java": "r", "Python": "b", "R": "g"}

    for longitude, latitude, language in get_cities('cities.csv'):
        plots[language][0].append(longitude)
        plots[language][1].append(latitude)

    # create a scatter series for each language
    for language, (x, y) in plots.items():
        plt.scatter(x, y, color=colors[language], marker=markers[language],
                    label=language, zorder=10)

    plot_state_borders()  # assume we have a function that does this

    plt.legend(loc=0)  # let matplotlib choose the location
    plt.axis([-130, -60, 20, 55])  # set the axes
    plt.title("Favorite Programming Languages")
    plt.show()


if __name__ == '__main__':
    plot_cities()
