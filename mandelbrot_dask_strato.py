from multiprocessing import Pool
from dask import delayed
from dask.distributed import Client, LocalCluster
import dask
import numpy as np
import time
import statistics
from numba import njit

@njit(fastmath=True)
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

@njit(fastmath=True)
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

def mandelbrot_dask(x_min, x_max, y_min, y_max, N, max_iter, n_chunks=None):
    chunk_size = max(1, N // n_chunks)
    tasks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(delayed(mandelbrot_chunk)(row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    
    parts = dask.compute(*tasks)
    return np.vstack(parts)
    
if __name__ == "__main__":    
    # initialize parameters
    x_min, x_max = -2, 1
    y_min, y_max = -1.5, 1.5
    max_iter = 100
    grid_sizes = ['1024', '2048', '4096', '8192']
    execution_times = []
    n_workers = 4
    
    client = Client("tcp://10.92.0.135:8786")
    client.run(lambda: mandelbrot_chunk(0, 8, 8, x_min, x_max, y_min, y_max, 10)) # warm up workers

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
        
        for chunk_mult in [1, 2, 4, 8, 16, 32, 64, 128]:
            chunks = chunk_mult
            dask_times = []
            
            for i in range(3):
                time_start = time.perf_counter()
                result = mandelbrot_dask(x_min, x_max, y_min, y_max, width, max_iter, chunks)
                dask_times.append(time.perf_counter() - time_start)
            dask_times_median = statistics.median(dask_times)
            speedup = median_serial_time / dask_times_median
            lif = n_workers * dask_times_median / median_serial_time - 1
            print(f"Chunks: {chunks}, Median Dask execution time: {dask_times_median:.4f} seconds, Speedup: {speedup:.2f}x, LIF: {lif:.2f}")
    
    client.close()