import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import pickle

import numpy as np
from scipy.optimize import curve_fit

with open("{0}/dicts/fit_funcs.pkl".format(os.environ["CD_CODE_DIR"]), "rb") as f:
    fit_funcs_dict = pickle.load(f)

FIT_PTS = 10000  # just needs to be much larger than agp_order to ensure no fine tuning


def fit_FE_universal_coeffs(
    agp_order, AGPtype, min_freq, max_freq, omega0, print_err=False
):
    """Calculate the coefficients from trying to fit -1/x by some basis choice of odd
    polynomials in the window [window_start, window_end]
    Parameters:
        agp_order (int):        Order of the AGP, i.e. number of odd polynomials
                                to attempt to fit 1/x with
        AGPtype (str):          Type of approximate AGP to construct, cannot be
                                'krylov' since this always depends on system
                                details and cannot be universal
        min_freq (float):       Point to start the fitting window of 1/x, must be > 0
        max_freq (float):       Point to end the fitting window of 1/x, assumed to
                                be > min_freq
        omega0 (float):         Single-particle reference frequency omega0, which
                                sets the scale of argument of Bessel functions
        print_err (bool):       Whether or not to print squared error per datapoint
    """
    # assuming H rescaled by 1 / max_freq
    x = np.linspace(min_freq / max_freq, 1, FIT_PTS)
    if "_alt" in AGPtype:
        y = -np.ones_like(x)
    else:
        y = -1 / x
    fit_func = fit_funcs_dict["bessel"]
    opt_coeffs, _ = curve_fit(fit_func, x / omega0, y, p0=np.zeros(agp_order))
    if print_err:
        fit_poly = fit_func(x, *opt_coeffs)
        error = np.sum((fit_poly - y) ** 2) / FIT_PTS
    return opt_coeffs
