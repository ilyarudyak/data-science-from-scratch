from collections import Counter, namedtuple
from linear_algebra import distance
from plot_state_borders import plot_state_borders
from stats import mean
import math
import random
import matplotlib.pyplot as plt
import csv


def get_cities(filename):
    with open(filename) as f:
        f_csv = csv.reader(f)
        headings = next(f_csv)
        City = namedtuple('City', headings)
        return [City(float(r[0]), float(r[1]), r[2]) for r in f_csv]
cities = get_cities('cities.csv')

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


def knn_classify2(k, cities, new_point):
    by_distance = sorted(cities,
                         key=lambda city:
                         distance((city.longitude, city.latitude), new_point))

    k_nearest_labels = [label for _, _, label in by_distance[:k]]

    return majority_vote(k_nearest_labels)


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


def classify_and_plot_grid(k=1):
    plots = {"Java": ([], []), "Python": ([], []), "R": ([], [])}
    markers = {"Java": "o", "Python": "s", "R": "^"}
    colors = {"Java": "r", "Python": "b", "R": "g"}

    for longitude in range(-130, -60):
        for latitude in range(20, 55):
            predicted_language = knn_classify2(k, cities, [longitude, latitude])
            plots[predicted_language][0].append(longitude)
            plots[predicted_language][1].append(latitude)

    # create a scatter series for each language
    for language, (x, y) in plots.items():
        plt.scatter(x, y, color=colors[language], marker=markers[language],
                    label=language, zorder=0)

    plot_state_borders()  # assume we have a function that does this

    plt.legend(loc=0)  # let matplotlib choose the location
    plt.axis([-130, -60, 20, 55])  # set the axes
    plt.title(str(k) + "-Nearest Neighbor Programming Languages")
    plt.show()


def choose_k():
    cities = get_cities('cities.csv')

    for k in [1, 3, 5, 7]:
        num_correct = 0

        for longitude, latitude, actual_language in cities:

            other_cities = [other_city
                            for other_city in cities
                            if other_city != (longitude, latitude, actual_language)]

            predicted_language = knn_classify2(k, other_cities, [longitude, latitude])

            if predicted_language == actual_language:
                num_correct += 1

        print(k, "neighbor[s]:", num_correct, "correct out of", len(cities))


if __name__ == '__main__':
    classify_and_plot_grid(k=1)
