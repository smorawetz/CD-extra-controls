import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import pickle

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import roots_chebyu

with open("{0}/dicts/fit_funcs.pkl".format(os.environ["CD_CODE_DIR"]), "rb") as f:
    fit_funcs_dict = pickle.load(f)

FIT_PTS = 10000  # just needs to be much larger than agp_order to ensure no fine tuning


def make_x_grid(min_freq, max_freq, num_pts=FIT_PTS):
    """Make a grid of x values for fitting"""
    return np.linspace(min_freq, max_freq, num_pts)
    # xpts, _ = roots_chebyu(num_pts)
    # keep_inds = np.where((max_freq * xpts >= min_freq))[0]
    # return xpts[keep_inds] * max_freq


def fit_universal_coeffs(agp_order, AGPtype, min_freq, max_freq, print_err=False):
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
        print_err (bool):       Whether or not to print squared error per datapoint
    """
    # assuming H rescaled by 1 / max_freq
    x = make_x_grid(min_freq / max_freq, 1.0, FIT_PTS)
    if "_alt" in AGPtype:
        y = -np.ones_like(x)
    else:
        y = -1 / x
    fit_func = fit_funcs_dict[AGPtype]
    opt_coeffs, _ = curve_fit(fit_func, x, y, p0=np.zeros(agp_order))
    if print_err:
        fit_poly = fit_func(x, *opt_coeffs)
        error = np.sum((fit_poly - y) ** 2) / FIT_PTS
        print(error)
    return opt_coeffs


def fit_universal_coeffs_regularized(
    agp_order, AGPtype, min_freq, max_freq, print_err=False
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
        print_err (bool):       Whether or not to print squared error per datapoint
    """
    # assuming H rescaled by 1 / max_freq
    x = make_x_grid(0, 1, FIT_PTS)
    y = -x / (x**2 + (min_freq / max_freq) ** 2)
    fit_func = fit_funcs_dict[AGPtype]
    opt_coeffs, _ = curve_fit(fit_func, x, y, p0=np.zeros(agp_order))
    if print_err:
        fit_poly = fit_func(x, *opt_coeffs)
        error = np.sum((fit_poly - y) ** 2) / FIT_PTS
        print(error)
    return opt_coeffs


def fit_universal_coeffs_no_rescale(
    agp_order, AGPtype, min_freq, max_freq, print_err=False
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
        print_err (bool):       Whether or not to print squared error per datapoint
    """
    x = make_x_grid(min_freq, max_freq, FIT_PTS)
    if "_alt" in AGPtype:
        y = -np.ones_like(x)
    else:
        y = -1 / x
    fit_func = fit_funcs_dict[AGPtype]
    opt_coeffs, _ = curve_fit(fit_func, x, y, p0=np.zeros(agp_order))
    if print_err:
        fit_poly = fit_func(x, *opt_coeffs)
        error = np.sum((fit_poly - y) ** 2) / FIT_PTS
        print(error)
    return opt_coeffs
