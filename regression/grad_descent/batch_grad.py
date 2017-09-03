import math
import random


def minimize_batch(target_fn, gradient_fn, x, y, theta_0, tolerance=0.01):
    theta = theta_0
    steps = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    value = target_fn(x, y, theta)

    while True:
        grad = gradient_fn(x, y, theta)
        next_thetas = [[theta[0] - step * grad[0], theta[1] - step * grad[1]] for step in steps]

        values = [target_fn(x, y, nt) for nt in next_thetas]
        next_theta = min(next_thetas, key=lambda t: target_fn(x, y, t))
        next_value = target_fn(x, y, next_theta)
        # print('next_thetas', next_thetas)
        # print('values', values)
        # print(next_theta, next_value)
        # print()

        if abs(value - next_value) < tolerance:
            break

        theta, value = next_theta, next_value

    return theta


def distance(u, v):
    return math.sqrt(sum([(ui - vi) ** 2 for ui, vi in zip(u, v)]))


def squared_error_batch(x, y, theta):
    alpha, beta = theta
    return sum([(yi - beta * xi - alpha) ** 2 for xi, yi in zip(x, y)])


def squared_error_gradient_batch(x, y, theta):
    alpha, beta = theta
    return [sum([-2 * (yi - beta * xi - alpha) for xi, yi in zip(x, y)]),
            sum([-2 * (yi - beta * xi - alpha) * xi for xi, yi in zip(x, y)])]


if __name__ == '__main__':
    random.seed(42)
    x = list(range(3))
    y = [2 * xi + 1 for xi in x]

    theta = [.5, .5]
    alpha, beta = minimize_batch(squared_error_batch,
                                 squared_error_gradient_batch,
                                 x,
                                 y,
                                 theta,
                                 0.000001)

    print("alpha", alpha)
    print("beta", beta)
