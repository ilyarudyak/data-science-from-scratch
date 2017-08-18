from collections import Counter
from linear_algebra import distance, vector_subtract, scalar_multiply
from functools import reduce
import math, random
import gradient_descent_sol


def sum_of_squares(v):
    """computes the sum of squared elements in v"""
    return sum(v_i ** 2 for v_i in v)


def difference_quotient(f, x, h):
    return (f(x + h) - f(x)) / h


def partial_difference_quotient(f, v, i, h):
    # add h to just the i-th element of v
    w = [v_j + (h if j == i else 0)
         for j, v_j in enumerate(v)]

    return (f(w) - f(v)) / h


def estimate_gradient(f, v, h=0.00001):
    """
        just calculate gradient using
        partial_difference_quotient()
    """
    return [partial_difference_quotient(f, v, i, h)
            for i, _ in enumerate(v)]


def step(v, direction, step_size):
    """
        this is the main update of gradient descent:
        v := v - a * dv
        direction here is gradient: dv
    """
    return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v, direction)]


def safe(f):
    """define a new function that wraps f and return it"""

    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')  # this means "infinity" in Python

    return safe_f


def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]


#
#
# minimize / maximize batch
#
#


def minimize_batch(cost_fn, gradient_fn, v0, tolerance=0.000001):
    """use gradient descent to find theta that minimizes target function"""

    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    v = v0  # set v to initial value
    cost_fn = safe(cost_fn)  # safe version of target_fn
    cost = cost_fn(v)  # value we're minimizing

    while True:
        gradient = gradient_fn(v)

        # choose next v using v := v + a * dv
        next_vs = [step(v, gradient, -step_size) for step_size in step_sizes]
        next_v = min(next_vs, key=cost_fn)

        # stop if we're "converging"
        next_cost = cost_fn(next_v)
        if abs(cost - next_cost) < tolerance:
            return v
        else:
            v, cost = next_v, next_cost


if __name__ == '__main__':

    print("using minimize_batch")

    v = [random.randint(-10, 10) for i in range(3)]

    v = minimize_batch(sum_of_squares, sum_of_squares_gradient, v, tolerance=.0001)

    print("minimum v", v)
    print("minimum value", sum_of_squares(v))
