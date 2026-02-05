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