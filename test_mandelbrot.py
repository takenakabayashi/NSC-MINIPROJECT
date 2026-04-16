from numba import njit
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