import numpy as np
from scipy.special import chebyt


def std_fit_func(x, *alphas):
    """Return the value of the standard odd polynomials fitting function
    (with coefficients alpha) for a particular choice of these coefficients
    Parameters:
        x (np.ndarray):     Point at which to evaluate the function
        *alphas (float):    Coefficients of the odd polynomials
    """
    y = 0
    for n, alpha in enumerate(alphas):
        y += alpha * x ** (2 * n + 1)
    return y


def std_fit_func_alt(x, *alphas):
    """Return the value of the standard odd polynomials fitting function
    (with coefficients alpha) for a particular choice of these coefficients
    Parameters:
        x (np.ndarray):     Point at which to evaluate the function
        *alphas (float):    Coefficients of the odd polynomials
    """
    y = 0
    for n, alpha in enumerate(alphas):
        y += alpha * x ** (2 * n + 2)
    return y


def cheby_fit_func(x, *coeffs):
    """Return the value of the fitting function, when fit with Chebyshev
    polynomials. Note that this only works well assuming H has been rescaled
    so that the max frequency is 1
    Parameters:
        x (np.ndarray):     Point at which to evaluate the function
        *coeffs (float):    Coefficients of the Chebyshev polynomials
    """
    y = 0
    for n, coeff in enumerate(coeffs):
        y += coeff * chebyt(2 * n + 1)(x)
    return y


def cheby_fit_func_alt(x, *coeffs):
    """Return the value of the fitting function, when fit with Chebyshev
    polynomials. Note that this only works well assuming H has been rescaled
    so that the max frequency is 1
    Parameters:
        x (np.ndarray):     Point at which to evaluate the function
        *coeffs (float):    Coefficients of the Chebyshev polynomials
    """
    y = 0
    for n, coeff in enumerate(coeffs):
        y += coeff * chebyt(2 * n + 1)(x) * x
    return y
