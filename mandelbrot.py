import numpy as np
import matplotlib.pyplot as plt
import time

max_iter = 100

def mandelbrot_point(c):
    """
    Determines if a complex number c is in the Mandelbrot set.
    
    :param c: Complex number to test if within Mandelbrot set
    """
    z = 0
    # Loop through number of iterations
    for n in range(max_iter):
        if(abs(z) > 2): # Check if z stays bounded
            return n # Return current number of iterations
        z = z*z + c # Update z using the Mandelbrot formula
    return max_iter # Return max iterations if in set

def compute_mandelbrot(x_min, x_max, y_min, y_max, width, height):
    """
    Compute Mandelbrot Set for a given range of x and y values.
    
    :param x_min: Minimum x value of the complex plane
    :param x_max: Maximum x value of the complex plane
    :param y_min: Minimum y value of the complex plane
    :param y_max: Maximum y value of the complex plane
    :param width: Number of points along the x-axis
    :param height: Number of points along the y-axis
    """
    
    # Compute evenly spaced values for x_min to x_max and y_min to y_max with 
    widthArr = np.linspace(x_min, x_max, width)
    heightArr = np.linspace(y_min, y_max, height)
    
    # Create grid of complex numbers from the x and y values
    complex_grid = np.array([[complex(x, y) for x in widthArr] for y in heightArr])
    
    # Initial array to hold Mandelbrot set results
    mandelbrot_set = np.zeros(complex_grid.shape, dtype=int)
    
    # Loop through each complex number and determine if it is in the Mandelbrot set
    for i, row in enumerate(complex_grid):
        for j, c in enumerate(row):
            result = mandelbrot_point(c)
            mandelbrot_set[i, j] = result
            
    return mandelbrot_set

def visualize_mandelbrot(mandelbrot_set, x_min, x_max, y_min, y_max, colormap):
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
    plt.title("Mandelbrot Set")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.show()
    
if __name__ == "__main__":
    # Measure start execution time
    start_time = time.time() 
    
    # x_min to x_max and y_min to y_max define the area of the complex plane to visualize
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    # width and height define the resolution of the output image
    width = 1024
    height = 1024
    
    # Color map for visualization 
    colormap = 'inferno'
    
    # Compute the Mandelbrot set for the specified area
    result = compute_mandelbrot(x_min, x_max, y_min, y_max, width, height)
    
    # Compute end time and print execution time
    end_time = time.time() - start_time
    print(f"Execution time: {end_time:.2f} seconds")
    
    # Visualize Mandelbrot Set
    visualize_mandelbrot(result, x_min, x_max, y_min, y_max, colormap)