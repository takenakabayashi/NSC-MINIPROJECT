import time, statistics

def benchmark(func, *args, n_runs):
    """
    Benchmarking function, which returns median of n_runs
    """
    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)
    median_time = statistics.median(times)
    print(f"Median: {median_time:.4f}s "
          f"(min={min(times):.4f}, max={max(times):.4f})")
    
    return median_time, result