import sys

from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import yaml
import cProfile, pstats
import os
import time
import statistics
from numba import njit

os.environ["LINE_PROFILE"] = '1'

@njit(cache=True)
def mandelbrot_point(c_real, c_img, max_iter):
    """
    Determines if a complex number c is in the Mandelbrot set.
    
    :param c: Complex number to test if within Mandelbrot set
    """
    z_real = z_img = 0.0
    # Loop through number of iterations
    for n in range(max_iter):
        z_real_squared = z_real * z_real
        z_img_squared = z_img * z_img
        if z_real_squared + z_img_squared > 4.0: # Check if z stays bounded
            return n # Return current number of iterations
        z_img = 2.0 * z_real * z_img + c_img # Update
        z_real = z_real_squared - z_img_squared + c_real # Update real part of z using the Mandelbrot formula
    return max_iter # Return max iterations if in set

@njit(cache=True)
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    output_grid = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_img = y_min + (r + row_start) * dy
        for col in range(N):
            c_real = x_min + col * dx
            output_grid[r, col] = mandelbrot_point(c_real, c_img, max_iter)
    return output_grid

def mandelbrot_serial(x_min, x_max, y_min, y_max, N, max_iter):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)

def _worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(x_min, x_max, y_min, y_max, N, max_iter, num_processes, n_chunks=None, pool=None):
    if n_chunks is None:
        n_chunks = num_processes
    chunk_size = max(1, N // n_chunks)
    chunks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    if pool is not None:
        return np.vstack(pool.map(_worker, chunks))
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    with Pool(processes=num_processes) as pool:
        pool.map(_worker, tiny)
        times_parallel = []
        for i in range(3):
            time_start = time.perf_counter()
            parts = np.vstack(pool.map(_worker, chunks))
            times_parallel.append(time.perf_counter() - time_start)
    time_median_parallel = statistics.median(times_parallel)
    print(f" ")
        
    return parts, time_median_parallel

def visualize_mandelbrot(mandelbrot_set, title, x_min, x_max, y_min, y_max, colormap):
    """
    Function to visualize Mandelbrot Set with a colormap
    
    :param mandelbrot_set: 2D array of Mandelbrot Set results
    :param x_min: Minimum x value of the complex plane
    :param x_max: Maximum x value of the complex plane
    :param y_min: Minimum y value of the complex plane
    :param y_max: Maximum y value of the complex plane
    :param colormap: Colormap to use for visualization
    """
    plt.imshow(mandelbrot_set, extent=[x_min, x_max, y_min, y_max], cmap=colormap, origin='lower', aspect='equal')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.show()
    
def grid_size_plot(grid_sizes, times, title="Mandelbrot Performance Tracking"):
    """
    Function to display grid sizes vs execution times bar plot
    
    :param grid_sizes: Array of grid sizes
    :param times: Array of execution times
    """
    plt.plot(grid_sizes, times, color='blue', marker='o')
    plt.title(title)
    plt.xlabel("Grid size")
    plt.ylabel("Execution times")
    plt.show()
    
def run_profiler(func, *args):
    """
    Function to run cProfile profiler on a given function and display stats
    
    :param func: Function to profile
    :param args: Arguments to pass to the function being profiled
    """
    profiler = cProfile.Profile()
    profiler.enable()
    func(*args)
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Print top 10 functions
    
if __name__ == "__main__":
    # Load available regions from YAML file
    with open("mandelbrot_regions.yml", 'r') as f:
        yaml_file = yaml.full_load(f)
    
    regions_dict = yaml_file.get("Regions", {})
    
    # Display available regions and prompt user to select one
    print("Regions: ", list(regions_dict.keys()))
    region = input("Choose a region: ").lower().title()
    
    if(len(regions_dict[region]) > 0):
        region = regions_dict[region]
    
    # initialize parameters
    x_min, x_max = region['x_min'], region['x_max']
    y_min, y_max = region['y_min'], region['y_max']
    max_iter = region['max_iter']
    grid_sizes = ['256', '512', '1024', '2048', '4096']
    grid_sizes = ['1024']
    implementations = ['naive', 'numpy', 'numba', 'numba_dtype']
    execution_times = []
    show_plots = False
    do_profiling = False
    
    optimal_workers = 4
    
    if not do_profiling:
        
        for gr_size in grid_sizes:
            width, height = int(gr_size), int(gr_size)
            print(f"Computing Mandelbrot set for grid size: {gr_size}x{gr_size}...")
            serial_times = []
            for i in range(3): # run serial mandelbrot implementation first
                time_start = time.perf_counter()
                mandelbrot_serial(x_min, x_max, y_min, y_max, width, max_iter)
                serial_times.append(time.perf_counter() - time_start)
            median_serial_time = statistics.median(serial_times)
            print(f"Median serial execution time: {median_serial_time:.4f} seconds")
            
            for mult in [1, 2, 4, 8, 16]:
                chunks = mult * optimal_workers
                result, time_median_parallel = mandelbrot_parallel(x_min, x_max, y_min, y_max, width, max_iter, optimal_workers, n_chunks=chunks)
                
                if(show_plots):
                    visualize_mandelbrot(result, f"Mandelbrot Set ({gr_size}x{gr_size})", x_min, x_max, y_min, y_max, 'inferno')
                    
                speedup = median_serial_time / time_median_parallel
                lif = optimal_workers * time_median_parallel / median_serial_time - 1
                print(f"Grid size: {gr_size}x{gr_size}, Workers: {optimal_workers}, Chunks: {chunks}, Parallel time: {time_median_parallel:.4f} seconds, Speedup: {speedup:.2f}x, Efficiency: {speedup/optimal_workers*100:.0f}%, LIF: {lif:.2f}")
    else:
        gr_size = 512
        width, height = gr_size, gr_size
        print(f"Profiling implementation for grid size: {gr_size}x{gr_size}...")
        run_profiler(mandelbrot_parallel, x_min, x_max, y_min, y_max, width, max_iter, 4)