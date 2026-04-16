from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import yaml
import time
import statistics
from numba import njit

@njit(cache=True)
def mandelbrot_point(c_real: float, c_img: complex, max_iter: int) -> int:
    """Compute escape count for a complex point.
    
    Iterates: :math:`z_{n+1} = z_n^2 + c` starting with :math:`z_0 = 0` and counts how many iterations it takes for the magnitude of :math:`z` to exceed 2. If it does not exceed 2 within `max_iter`, returns `max_iter`.

    Parameters:
        c_real (float): The real part of the complex coordinate
        c_img (complex): The imaginary part of the complex coordinate
        max_iter (int): Maximum number of iterations to test for Mandelbrot set

    Returns:
        int: First iteration count where the point escapes, or max_iter if it does not escape
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
def mandelbrot_chunk(row_start: int, row_end: int, N: int, x_min: float, x_max: float, y_min: float, y_max: float, max_iter: int) -> np.ndarray:
    """Generates the chunk that the Mandelbrot function is computed within

    Parameters:
        row_start (int): Starting row index for the chunk
        row_end (int): End row index for the chunk
        N (int): Number of points along each axis (grid size)
        x_min (float): Minimum x value of the complex plane
        x_max (float): Maximum x value of the complex plane
        y_min (float): Minimum y value of the complex plane
        y_max (float): Maximum y value of the complex plane
        max_iter (int): Maximum number of iterations to test for Mandelbrot set

    Returns:
        np.ndarray: 2D numpy array containing the escape counts for the specified chunk of the Mandelbrot set
    """
    output_grid = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_img = y_min + (r + row_start) * dy
        for col in range(N):
            c_real = x_min + col * dx
            output_grid[r, col] = mandelbrot_point(c_real, c_img, max_iter)
    return output_grid

def mandelbrot_serial(x_min: float, x_max: float, y_min: float, y_max: float, N: int, max_iter: int) -> np.ndarray:
    """Helper function to perform Mandelbrot in serial

    Parameters:
        x_min (float): Minimum x value of the complex plane
        x_max (float): Maximum x value of the complex plane
        y_min (float): Minimum y value of the complex plane
        y_max (float): Maximum y value of the complex plane
        N (int): Number of points along each axis (grid size)
        max_iter (int): Maximum number of iterations to test for Mandelbrot set

    Returns:
        np.ndarray: 2D numpy array containing the escape counts for the specified chunk of the Mandelbrot set
    """
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)

def _worker(args):
    """Worker function to generate chunks for performing multiprocessing Mandelbrot

    Args:
        args (_type_): Arguments for mandelbrot_chunk function

    Returns:
        _type_: Output of mandelbrot_chunk function
    """
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(x_min: float, x_max: float, y_min: float, y_max: float, N: int, max_iter: int, num_processes: int, n_chunks=None, pool=None) -> tuple[np.ndarray, float]:
    """Performs parallel Mandelbrot using either num_processes or n_chunks

    Parameters:
        x_min (float): Minimum x value of the complex plane
        x_max (float): Maximum x value of the complex plane
        y_min (float): Minimum y value of the complex plane
        y_max (float): Maximum y value of the complex plane
        N (int): Number of points along each axis (grid size)
        max_iter (int): Maximum number of iterations to test for Mandelbrot set
        num_processes (int): Number of processes to use for parallel execution
        n_chunks (_type_, optional): Number of chunks to divide the work into. Defaults to None.
        pool (_type_, optional): Process pool for parallel execution. Defaults to None.

    Returns:
        tuple[np.ndarray, float]: A tuple containing the generated Mandelbrot set and the median execution time.
    """
    if n_chunks is None:
        n_chunks = num_processes
    chunk_size = max(1, N // n_chunks)
    chunks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    if pool is not None:
        return np.vstack(pool.map(_worker, chunks)), 0.0
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    with Pool(processes=num_processes) as pool:
        pool.map(_worker, tiny)
        times_parallel = []
        for i in range(3):
            time_start = time.perf_counter()
            parts = np.vstack(pool.map(_worker, chunks))
            times_parallel.append(time.perf_counter() - time_start)
    time_median_parallel = statistics.median(times_parallel)
        
    return parts, time_median_parallel

def visualize_mandelbrot(mandelbrot_set: np.ndarray, title: str, x_min: float, x_max: float, y_min: float, y_max: float, colormap: str) -> None:
    """Function to visualize a Mandelbrot set using a Pyplot plot

    Parameters:
        mandelbrot_set (np.ndarray): 2D numpy array containing the escape counts for the specified chunk of the Mandelbrot set
        title (str): Title for the plot
        x_min (float): Minimum x value of the complex plane
        x_max (float): Maximum x value of the complex plane
        y_min (float): Minimum y value of the complex plane
        y_max (float): Maximum y value of the complex plane
        colormap (str): Colormap to use for visualization
    """
    plt.imshow(mandelbrot_set, extent=[x_min, x_max, y_min, y_max], cmap=colormap, origin='lower', aspect='equal') # type: ignore
    plt.colorbar()
    plt.title(title)
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.show()
    
def grid_size_plot(grid_sizes: list, times: list, title="Mandelbrot Performance Tracking") -> None:
    """Function to show a plot comparing grid sizes to execution times

    Parameters:
        grid_sizes (list): List of grid sizes
        times (list): List of execution times
        title (str, optional): Title of plot. Defaults to "Mandelbrot Performance Tracking".
    """
    plt.plot(grid_sizes, times, color='blue', marker='o')
    plt.title(title)
    plt.xlabel("Grid size")
    plt.ylabel("Execution times")
    plt.show()
    
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
    x_min, x_max = float(region['x_min']), float(region['x_max']) #type: ignore
    y_min, y_max = float(region['y_min']), float(region['y_max']) #type: ignore
    max_iter = int(region['max_iter']) #type: ignore
    grid_sizes = ['256', '512', '1024', '2048', '4096']
    grid_sizes = ['1024']
    implementations = ['naive', 'numpy', 'numba', 'numba_dtype']
    execution_times = []
    show_plots = False
    
    optimal_workers = 6
    
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
        
        for mult in [1, 2, 4, 8, 16, 32, 64]:
            chunks = mult * optimal_workers
            result, time_median_parallel = mandelbrot_parallel(x_min, x_max, y_min, y_max, width, max_iter, optimal_workers, n_chunks=chunks)
            
            if(show_plots):
                visualize_mandelbrot(result, f"Mandelbrot Set ({gr_size}x{gr_size})", x_min, x_max, y_min, y_max, 'inferno')
                
            speedup = median_serial_time / time_median_parallel
            lif = optimal_workers * time_median_parallel / median_serial_time - 1
            print(f"Grid size: {gr_size}x{gr_size}, Workers: {optimal_workers}, Chunks: {chunks}, Parallel time: {time_median_parallel:.4f} seconds, Speedup: {speedup:.2f}x, Efficiency: {speedup/optimal_workers*100:.0f}%, LIF: {lif:.2f}")