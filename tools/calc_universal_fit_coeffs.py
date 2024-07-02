import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import chebyt

FIT_PTS = 100  # just needs to be much larger than agp_order to ensure no fine tuning


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


## NOTE: currently only do linearly spaced fit, can probably improve by playing with this
def fit_universal_coeffs(fit_func, agp_order, min_freq, max_freq):
    """Calculate the coefficients from trying to fit -1/x by some basis choice of odd
    polynomials in the window [window_start, window_end]
    Parameters:
        fit_func (function):    Function to use to fit 1/x, must take in x and coeffs
        agp_order (int):        Order of the AGP, i.e. number of odd polynomials
                                to attempt to fit 1/x with
        min_freq (float):       Point to start the fitting window of 1/x, must be > 0
        max_freq (float):       Point to end the fitting window of 1/x, assumed to
                                be > min_freq
    """
    # assuming H rescaled by 1 / max_freq
    x = np.linspace(min_freq / max_freq, 1, FIT_PTS)
    y = -1 / x
    print(x, y)
    opt_coeffs, _ = curve_fit(fit_func, x, y, p0=np.zeros(agp_order))
    return opt_coeffs
