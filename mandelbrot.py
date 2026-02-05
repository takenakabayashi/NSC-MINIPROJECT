import numpy as np

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

def compute_mandelbrot(x_min, x_max, y_min, y_max):
    """
    Compute Mandelbrot Set for a given range of x and y values.
    
    :param x_min: Minimum x value of the complex plane
    :param x_max: Maximum x value of the complex plane
    :param y_min: Minimum y value of the complex plane
    :param y_max: Maximum y value of the complex plane
    """
    
    # Compute evenly spaced values for x_min to x_max and y_min to y_max
    width = np.linspace(x_min, x_max, max_iter)
    height = np.linspace(y_min, y_max, max_iter)
    
    # Create grid of complex numbers from the x and y values
    complex_grid = np.array([[complex(x, y) for x in width] for y in height])
    
    # Initial array to hold Mandelbrot set results
    mandelbrot_set = np.zeros(complex_grid.shape, dtype=int)
    
    # Loop through each complex number and determine if it is in the Mandelbrot set
    for i, row in enumerate(complex_grid):
        for j, c in enumerate(row):
            result = mandelbrot_point(c)
            mandelbrot_set[i, j] = result
            
    return mandelbrot_set
    
if __name__ == "__main__":
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    result = compute_mandelbrot(x_min, x_max, y_min, y_max)
    print(result)