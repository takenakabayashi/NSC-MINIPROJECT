from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import yaml
from numba import njit

def mandelbrot(x_min, x_max, y_min, y_max, N, max_iter, TAU=0.01):
    x = np.linspace(x_min, x_max, N)
    y = np.linspace(y_min, y_max, N)
    C64 = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
    C32 = C64.astype(np.complex64)
    z32 = np.zeros_like(C32, dtype=np.complex64)
    z64 = np.zeros_like(C64, dtype=np.complex128)
    
    diverge = np.full((N, N), max_iter, dtype=np.int32)
    active = np.ones((N, N), dtype=bool)
    
    for i in range(max_iter):
        if not active.any(): break
        z32[active] = z32[active]**2 + C32[active]
        z64[active] = z64[active]**2 + C64[active]
        diff = (np.abs(z32.real.astype(np.float64) - z64.real) + np.abs(z32.imag.astype(np.float64) - z64.imag))
        newly = active & (diff > TAU)
        diverge[newly] = i
        active[newly] = False
        
    return diverge


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
    grid_sizes = ['512']
    TAU = 0.01
    show_plots = False
    
    for gr_size in grid_sizes:
        width, height = int(gr_size), int(gr_size)
        diverge = mandelbrot(x_min, x_max, y_min, y_max, width, max_iter, TAU)
        
        print(diverge[diverge < max_iter].size, "points diverged before max iterations.")
        print("Fraction of points that diverged: ", diverge[diverge < max_iter].size / (width * height))
        
        plt.imshow(diverge, cmap='inferno', extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='equal')
        plt.colorbar(label='Divergence Iteration')
        plt.title(f"Mandelbrot Set Divergence (TAU: {TAU}) - Grid Size: {gr_size}x{gr_size}")
        plt.show()