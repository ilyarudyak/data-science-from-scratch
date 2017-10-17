import numpy as np
import math
import random
import time


def minimize_batch(target_fn, gradient_fn, x, y, theta_0, tolerance=0.01):
    theta = theta_0
    learning_rates = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    value = target_fn(x, y, theta)

    while True:
        grad = gradient_fn(x, y, theta)
        next_thetas = [[theta[0] - rate * grad[0], theta[1] - rate * grad[1]] for rate in learning_rates]
        print(next_thetas)

        next_theta = min(next_thetas, key=lambda t: target_fn(x, y, t))
        next_value = target_fn(x, y, next_theta)

        if abs(value - next_value) < tolerance:
            break

        theta, value = next_theta, next_value

    return theta


def minimize_batch_numpy(target_fn, gradient_fn, x, y, theta_0, tolerance=0.01):
    theta = theta_0.reshape(1, 2)
    learning_rates = np.array([100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]).reshape(8, 1)
    value = target_fn(x, y, theta)

    while True:
        grad = gradient_fn(x, y, theta).reshape(1, 2)
        next_thetas = theta - learning_rates * grad

        next_theta = min(next_thetas, key=lambda t: target_fn(x, y, t))
        next_value = target_fn(x, y, next_theta)

        if abs(value - next_value) < tolerance:
            break

        theta, value = next_theta, next_value

    return theta


def minimize_batch_numpy2(target_fn, gradient_fn, x, y, theta_0, tolerance=0.01):
    theta = theta_0.reshape(1, 2)
    learning_rates = np.array([100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]).reshape(8, 1)
    value = target_fn(x, y, theta)

    while True:
        grad = gradient_fn(x, y, theta).reshape(1, 2)
        next_thetas = theta - learning_rates * grad

        next_theta = min(next_thetas, key=lambda t: target_fn(x, y, t))
        next_value = target_fn(x, y, next_theta)

        if abs(value - next_value) < tolerance:
            break

        theta, value = next_theta, next_value

    return theta


def distance(u, v):
    return math.sqrt(sum([(ui - vi) ** 2 for ui, vi in zip(u, v)]))


def distance_numpy(u, v):
    return np.sqrt(np.sum((u - v) ** 2))


def squared_error_batch(x, y, theta):
    alpha, beta = theta
    return sum([(yi - beta * xi - alpha) ** 2 for xi, yi in zip(x, y)])


def squared_error_batch_numpy(x, y, theta):
    alpha, beta = theta.squeeze()
    return np.sum((y - beta * x - alpha) ** 2)


def squared_error_batch_numpy2(x, y, theta):
    alpha, beta = theta.squeeze()
    return np.sum((y - beta * x - alpha) ** 2)


def squared_error_gradient_batch(x, y, theta):
    alpha, beta = theta
    return [sum([-2 * (yi - beta * xi - alpha) for xi, yi in zip(x, y)]),
            sum([-2 * (yi - beta * xi - alpha) * xi for xi, yi in zip(x, y)])]


def squared_error_gradient_batch_numpy(x, y, theta):
    alpha, beta = theta.squeeze()
    return np.array([np.sum(-2 * (y - beta * x - alpha)),
                     np.sum(-2 * (y - beta * x - alpha) * x)])


def test_batch_numpy(n=3):
    random.seed(42)
    x = np.arange(n)
    y = 2 * x + 1

    theta = np.array([.5, .5])

    tic = time.time()
    alpha, beta = minimize_batch_numpy(squared_error_batch_numpy,
                                       squared_error_gradient_batch_numpy,
                                       x,
                                       y,
                                       theta,
                                       0.00000001)
    toc = time.time()

    print("alpha={:.3f}, beta={:.3f}, time={:.2f}ms".format(alpha, beta, (toc - tic) * 1000))


if __name__ == '__main__':
    test_batch_numpy(67)

