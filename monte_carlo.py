import math
import time
import random
import statistics
from multiprocessing import Pool
import os

def plot_execution_times(execution_times):
    import matplotlib.pyplot as plt
    processes, times = zip(*execution_times)
    plt.figure(figsize=(10, 6))
    plt.plot(processes, times, marker='o')
    plt.title('Execution Time vs Number of Processes')
    plt.xlabel('Number of Processes')
    plt.ylabel('Execution Time (seconds)')
    plt.xticks(processes)
    plt.grid()
    plt.show()

def estimate_pi_chunk(num_samples):
    num_inside_circle = 0
    for i in range(num_samples):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if x*x + y*y <= 1:
            num_inside_circle += 1
    return num_inside_circle

def estimate_pi_parallel(num_samples, num_processes = 4):
    samples_per_process = num_samples // num_processes
    tasks = [samples_per_process] * num_processes
    with Pool(processes=num_processes) as pool:
        results = pool.map(estimate_pi_chunk, tasks)
    return 4 * sum(results) / num_samples

if __name__ == "__main__":
    num_samples = 10**6
    num_trials = 5
    
    total_median_execution_times = []
    
    for num_proc in range(1, os.cpu_count() + 1):
        execution_times = []
        
        for i in range(num_trials):
            start_time = time.perf_counter()
            pi_estimate = estimate_pi_parallel(num_samples, num_processes=num_proc)
            execution_times.append(time.perf_counter() - start_time)
        time_parallel = statistics.median(execution_times)
        print(f"Estimated Pi: {pi_estimate:.6f} (difference: {abs(math.pi - pi_estimate):.6f})")
        print(f"Execution Time (Parallel with {num_proc} processes): {time_parallel:.3f} seconds")
        total_median_execution_times.append((num_proc, time_parallel))
        
    plot_execution_times(total_median_execution_times)