from numba import njit
import numpy as np
import pytest

def mandelbrot_pixel(c: complex, max_iter: int) -> int:
    """Outputs max iteration of Mandelbrot for a given complex number

    Args:
        c (complex): Complex number to test if within Mandelbrot set
        max_iter (int): Maximum number of iterations to test for Mandelbrot set

    Returns:
        int: Max iteration count for Mandelbrot set
    """
    z = 0j
    for n in range(max_iter):
        if z.real*z.real + z.imag*z.imag > 4.0:
            return n
        z = z*z + c
    return max_iter

@njit(cache=True)
def mandelbrot_pixel_numba(c: complex, max_iter: int) -> int:
    """Outputs max iteration of Mandelbrot for a given complex number

    Args:
        c (complex): Complex number to test if within Mandelbrot set
        max_iter (int): Maximum number of iterations to test for Mandelbrot set

    Returns:
        int: Max iteration count for Mandelbrot set
    """
    z = 0j
    for n in range(max_iter):
        if z.real*z.real + z.imag*z.imag > 4.0:
            return n
        z = z*z + c
    return max_iter

KNOWN_CASES = [
    (0+0j, 100, 100), # origin so it never escapes
    (5.0+0j, 100, 1), # far outside, so it should escape on the first iteration
    (-2.5+0j, 100, 1), # left edge of the set
]

IMPLEMENTATIONS = [mandelbrot_pixel, mandelbrot_pixel_numba]

@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
@pytest.mark.parametrize("c, max_iter, expected", KNOWN_CASES)
def test_pixel_all(impl, c, max_iter, expected):
    assert impl(c, max_iter) == expected
    
    
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_pixel_boundary_strict_inequality(impl):
    assert impl(0+2j, 100) == 2
    
def test_naive_and_numba_agree_on_grid():
    N = 32
    max_iter = 100
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    
    naive_grid = np.empty((N, N), dtype=np.int32)
    numba_grid = np.empty((N, N), dtype=np.int32)
    
    for row in range(N):
        c_imag = y_min + row * (y_max - y_min) / N
        for col in range(N):
            c_real = x_min + col * (x_max - x_min) / N
            c = complex(c_real, c_imag)
            naive_grid[row, col] = mandelbrot_pixel(c, max_iter)
            numba_grid[row, col] = mandelbrot_pixel_numba(c, max_iter)
    
    np.testing.assert_array_equal(naive_grid, numba_grid)