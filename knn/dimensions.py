import random
from linear_algebra import distance
from stats import mean
import matplotlib.pyplot as plt


def random_point(dim):
    return [random.random() for _ in range(dim)]


def random_distances(dim, num_pairs):
    return [distance(random_point(dim), random_point(dim))
            for _ in range(num_pairs)]


dimensions = range(1, 101, 10)

avg_distances = []
min_distances = []

random.seed(0)
for dim in dimensions:
    distances = random_distances(dim, 100)  # 10,000 random pairs
    avg_distances.append(mean(distances))  # track the average
    min_distances.append(min(distances))  # track the minimum
    print('{:2d} {:.2f} {:.2f} {:.2f}'.format(dim, min(distances), mean(distances),
                                              min(distances) / mean(distances)))

# plt.plot(dimensions, avg_distances)
# plt.plot(dimensions, min_distances)

min_avg_ratio = [min_dist / avg_dist
                 for min_dist, avg_dist in zip(min_distances, avg_distances)]
plt.plot(dimensions, min_avg_ratio)
plt.show()
