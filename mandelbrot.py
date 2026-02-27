import sys

import line_profiler
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import yaml
import cProfile, pstats
import os
from numba import njit
from benchmark import benchmark

os.environ["LINE_PROFILE"] = '1'

#@line_profiler.profile
def mandelbrot_point(c, max_iter):
    """
    Determines if a complex number c is in the Mandelbrot set.
    
    :param c: Complex number to test if within Mandelbrot set
    """
    z = 0j
    # Loop through number of iterations
    for n in range(max_iter):
        if z.real*z.real + z.imag*z.imag > 4.0: # Check if z stays bounded
            return n # Return current number of iterations
        z = z*z + c # Update z using the Mandelbrot formula
    return max_iter # Return max iterations if in set

#@line_profiler.profile
def compute_mandelbrot_naive(x_min, x_max, y_min, y_max, width, height, max_iter):
    """
    Compute Mandelbrot Set for a given range of x and y values.
    
    :param x_min: Minimum x value of the complex plane
    :param x_max: Maximum x value of the complex plane
    :param y_min: Minimum y value of the complex plane
    :param y_max: Maximum y value of the complex plane
    :param width: Number of points along the x-axis
    :param height: Number of points along the y-axis
    :max_iter: Maximum number of iterations to determine if a point is in the Mandelbrot set
    """
    # Compute evenly spaced values for x_min to x_max and y_min to y_max
    widthArr = np.linspace(x_min, x_max, width)
    heightArr = np.linspace(y_min, y_max, height)
    
    complex_grid = [[complex(x, y) for x in widthArr] for y in heightArr] # Create grid of complex numbers from the x and y values
    mandelbrot_set = [[0 for _ in range(width)] for _ in range(height)] # Initial 2D array to hold Mandelbrot set results
    
    # Loop through each complex number and determine if it is in the Mandelbrot set
    for i, row in enumerate(complex_grid):
        for j, c in enumerate(row):
            mandelbrot_set[i][j] = mandelbrot_point(c, max_iter)
            
    return mandelbrot_set

def compute_mandelbrot_numpy(x_min, x_max, y_min, y_max, width, height, max_iter):
    """
    Compute Mandelbrot Set for a given range of x and y values.
    
    :param x_min: Minimum x value of the complex plane
    :param x_max: Maximum x value of the complex plane
    :param y_min: Minimum y value of the complex plane
    :param y_max: Maximum y value of the complex plane
    :param width: Number of points along the x-axis
    :param height: Number of points along the y-axis
    :max_iter: Maximum number of iterations to determine if a point is in the Mandelbrot set
    """
    # Compute evenly spaced values for x_min to x_max and y_min to y_max 
    widthArr = np.linspace(x_min, x_max, width)
    heightArr = np.linspace(y_min, y_max, height)
    
    X, Y = np.meshgrid(widthArr, heightArr)
    complex_grid = X + 1j * Y
    
    Z_set = np.zeros(complex_grid.shape, dtype=complex)
    # Initial array to hold Mandelbrot set results
    mandelbrot_set = np.zeros(complex_grid.shape, dtype=int)
    
    for i in range(max_iter):
        mask = np.abs(Z_set) <= 2
        Z_set[mask] = Z_set[mask]**2 + complex_grid[mask]
        mandelbrot_set[mask] += 1
            
    return mandelbrot_set

@njit
def compute_mandelbrot_numba(x_min, x_max, y_min, y_max, width, height, max_iter):
    """
    Compute Mandelbrot Set for a given range of x and y values.
    
    :param x_min: Minimum x value of the complex plane
    :param x_max: Maximum x value of the complex plane
    :param y_min: Minimum y value of the complex plane
    :param y_max: Maximum y value of the complex plane
    :param width: Number of points along the x-axis
    :param height: Number of points along the y-axis
    :max_iter: Maximum number of iterations to determine if a point is in the Mandelbrot set
    """
    # Compute evenly spaced values for x_min to x_max and y_min to y_max
    widthArr = np.linspace(x_min, x_max, width)
    heightArr = np.linspace(y_min, y_max, height)
    
    complex_grid = [[complex(x, y) for x in widthArr] for y in heightArr] # Create grid of complex numbers from the x and y values
    mandelbrot_set = [[0 for _ in range(width)] for _ in range(height)] # Initial 2D array to hold Mandelbrot set results
    
    # Loop through each complex number and determine if it is in the Mandelbrot set
    for i, row in enumerate(complex_grid):
        for j, c in enumerate(row):
            z = 0j
            n = 0
            while n < max_iter and z.real*z.real + z.imag*z.imag <= 4.0: # Check if z stays bounded
                z = z*z + c # Update z using the Mandelbrot formula
                n += 1
            mandelbrot_set[i][j] = n # Return current number of iterations or max_iter if in set
            
    return mandelbrot_set

@njit
def compute_mandelbrot_numba_dtype(x_min, x_max, y_min, y_max, width, height, max_iter, dtype=np.float64):
    """
    Compute Mandelbrot Set for a given range of x and y values.
    
    :param x_min: Minimum x value of the complex plane
    :param x_max: Maximum x value of the complex plane
    :param y_min: Minimum y value of the complex plane
    :param y_max: Maximum y value of the complex plane
    :param width: Number of points along the x-axis
    :param height: Number of points along the y-axis
    :max_iter: Maximum number of iterations to determine if a point is in the Mandelbrot set
    """
    # Compute evenly spaced values for x_min to x_max and y_min to y_max
    widthArr = np.linspace(x_min, x_max, width).astype(dtype)
    heightArr = np.linspace(y_min, y_max, height).astype(dtype)
    mandelbrot_set = np.zeros((height, width), dtype=np.int32) # Initial 2D array to hold Mandelbrot set results
    
    # Loop through each complex number and determine if it is in the Mandelbrot set
    for i, row in enumerate(mandelbrot_set):
        for j, c in enumerate(row):
            z = 0j
            n = 0
            c = widthArr[j] + 1j * heightArr[i] # Get complex number from grid
            while n < max_iter and z.real*z.real + z.imag*z.imag <= 4.0: # Check if z stays bounded
                z = z*z + c # Update z using the Mandelbrot formula
                n += 1
            mandelbrot_set[i][j] = n # Return current number of iterations or max_iter if in set
            
    return mandelbrot_set

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
    plt.imshow(mandelbrot_set, extent=[x_min, x_max, y_min, y_max], cmap=colormap)
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
    implementations = ['naive', 'numpy', 'numba', 'numba_dtype']
    execution_times = []
    show_plots = True
    do_profiling = False
    
    for impl in implementations:
        print(f"Running {impl} implementation...")
        
        if impl == 'naive':
            compute_func = compute_mandelbrot_naive
        elif impl == 'numpy':
            compute_func = compute_mandelbrot_numpy
        elif impl == 'numba':
            compute_func = compute_mandelbrot_numba
        elif impl == 'numba_dtype':
            compute_func = lambda *args: compute_mandelbrot_numba_dtype(*args, dtype=np.float32) # Use float32 for numba_dtype implementation
        
        if not do_profiling: 
            impl_execution_times = []
            
            for gr_size in grid_sizes:
                gr_size = int(gr_size)
                width, height = gr_size, gr_size
                print(f"Computing Mandelbrot set for grid size: {gr_size}x{gr_size}...")
        
                if impl == 'numba' or impl == 'numba_dtype':
                    _ = compute_func(x_min, x_max, y_min, y_max, width, height, max_iter) # Warm up run to compile numba function
                
                if impl == 'numba_dtype':
                    _ = compute_mandelbrot_numba_dtype(x_min, x_max, y_min, y_max, width, height, max_iter, dtype=np.float32) # Warm up run to compile numba_dtype function with float32
                    
                    result_32 = compute_mandelbrot_numba_dtype(x_min, x_max, y_min, y_max, width, height, max_iter, dtype=np.float32) # Run with float32
                    result_64 = compute_mandelbrot_numba_dtype(x_min, x_max, y_min, y_max, width, height, max_iter, dtype=np.float64) # Run with float64
                    
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    for ax, result, title in zip(axes, [result_32, result_64], ['Numba Dtype (float32)', 'Numba Dtype (float64)']):
                        ax.imshow(result, cmap='inferno')
                        ax.set_title(title)
                        ax.axis('off')
                    plt.savefig('precision_comparison.png')
                    
                    print(f"Max diff float32 vs float64: {np.max(np.abs(result_32 - result_64))}")
                    sys.exit()
                else:
                    med_time, result = benchmark(compute_func, x_min, x_max, y_min, y_max, width, height, max_iter, n_runs=3)
                    print(f"Execution time ({gr_size}x{gr_size}): {med_time:.4f} seconds")
                
                # Color map for visualization 
                colormap = 'inferno'
                
                # Visualize Mandelbrot Set
                if(show_plots):
                    visualize_mandelbrot(result, f"Mandelbrot Set ({gr_size}x{gr_size})", x_min, x_max, y_min, y_max, colormap)
                
                impl_execution_times.append(med_time)
            
            grid_size_plot(grid_sizes, impl_execution_times, f"{impl.capitalize()} implementation performance tracking")
        else:
            gr_size = 512
            width, height = gr_size, gr_size
            print(f"Profiling {impl} implementation for grid size: {gr_size}x{gr_size}...")
            run_profiler(compute_func, x_min, x_max, y_min, y_max, width, height, max_iter)