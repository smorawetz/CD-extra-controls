import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import numpy as np
from scipy.optimize import curve_fit

FIT_PTS = 100  # just needs to be much larger than agp_order to ensure no fine tuning


def fit_func(x, *alphas):
    """Return the value of the fitting function for a particular choice of alphas
    Parameters:
        x (np.ndarray):     Point at which to evaluate the function
        *alphas (float):    Coefficients of the odd polynomials
    """
    y = 0
    for n, alpha in enumerate(alphas):
        y += alpha * x ** (2 * n + 1)
    return y


## NOTE: currently only do linearly spaced fit, can probably improve by playing with this
def fit_universal_coeffs(agp_order, window_start, window_end):
    """Calculate the coefficients $\alpha_i$ obtained from trying to fit 1/x by odd
    polynomials in the window [window_start, window_end]
    Parameters:
        agp_order (int):        Order of the AGP, i.e. number of odd polynomials
                                to attempt to fit 1/x with
        window_start (float):   Point to start the fitting window of 1/x, must be > 0
        window_end (float):     Point to end the fitting window of 1/x, assumed to
                                be > window_start
    """
    x = np.linspace(window_start, window_end, FIT_PTS)
    y = 1 / x
    opt_alphas, _ = curve_fit(fit_func, x, y, p0=np.zeros(agp_order))
    return -opt_alphas  # negative sign since fitting 1/x by -\sum_n \alpha_n x^{2n+1}
