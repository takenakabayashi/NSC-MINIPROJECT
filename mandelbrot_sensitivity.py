from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import yaml
from numba import njit
from matplotlib.colors import LogNorm

def mandelbrot(x_min, x_max, y_min, y_max, N, max_iter):
    x = np.linspace(x_min, x_max, N)
    y = np.linspace(y_min, y_max, N)
    C = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
    eps32 = float(np.finfo(np.float32).eps)
    delta = np.maximum(eps32 * np.abs(C), 1e-10)
    
    def escape_count(C, max_iter):
        z = np.zeros_like(C, dtype=np.complex128)
        cnt = np.full(C.shape, max_iter, dtype=np.int32)
        esc = np.zeros(C.shape, dtype=bool)
        for i in range(max_iter):
            z[~esc] = z[~esc]**2 + C[~esc]
            newly = ~esc & (np.abs(z) > 2)
            cnt[newly] = i
            esc[newly] = True
        return cnt

    n_base = escape_count(C, max_iter).astype(float)
    n_perturbed = escape_count(C + delta, max_iter).astype(float)
    dn = np.abs(n_base - n_perturbed)
    kappa = np.where(n_base > 0, dn / (eps32 * n_base), np.nan)
    
    return kappa

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
    show_plots = False
    
    for gr_size in grid_sizes:
        width, height = int(gr_size), int(gr_size)
        kappa = mandelbrot(x_min, x_max, y_min, y_max, width, max_iter)
        
        vmax = np.nanpercentile(kappa, 99)
        cmap_k = plt.cm.hot.copy()
        cmap_k.set_bad(color='0.25')
        
        plt.imshow(kappa, extent=[x_min, x_max, y_min, y_max], cmap=cmap_k, origin='lower', aspect='equal', norm=LogNorm(vmin=1, vmax=vmax))
        plt.colorbar(label='Sensitivity (kappa)')
        plt.title('Condition number (kappa) of Mandelbrot Set')
        plt.show()