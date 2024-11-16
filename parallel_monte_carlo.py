import numpy as np
import matplotlib.pyplot as plt
import random
import time
import multiprocessing as mp

def in_mandelbrot_set(x, y, max_iter):
    """Determine if a complex number is part of the Mandelbrot set."""
    a1, b1 = 0, 0
    for _ in range(max_iter):
        if (a1 * a1 + b1 * b1) >= 4:
            return 0
        a, b = a1 * a1 - b1 * b1 + x, 2 * a1 * b1 + y
        a1, b1 = a, b
    return 1

def worker(args):
    x_min, x_max, y_min, y_max, samples, max_iter = args
    count_inside = 0
    for _ in range(samples):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        count_inside += in_mandelbrot_set(x, y, max_iter)
    return count_inside

def monte_carlo_multi(samples=2000000, max_iter=100, xrange=(-2, 1.5), yrange=(-1.5, 1.5), processes=4):
    x_min, x_max = xrange
    y_min, y_max = yrange
    region_area = (x_max - x_min) * (y_max - y_min)

    samples_per_process = samples // processes

    tasks = [(x_min, x_max, y_min, y_max, samples_per_process, max_iter) for _ in range(processes)] # we lose edges here with sample

    with mp.Pool(processes=processes) as pool:
        results = pool.map(worker, tasks)

    total_inside = sum(results)
    mandel_area = (total_inside / samples) * region_area

    return mandel_area

if __name__ == '__main__':
    start = time.time()
    area = monte_carlo_multi(samples=2000000, max_iter=100, processes=1)
    print(f"area: {area}")
    stop = time.time()
    print(f"Total time: {stop-start}")
