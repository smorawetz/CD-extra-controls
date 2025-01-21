import os
import sys

import pickle

import numpy as np
import scipy

from scipy.optimize import curve_fit

sys.path.append(os.environ["CD_CODE_DIR"])

from utils.file_IO import save_data_agp_coeffs
from utils.file_naming import make_data_dump_name, make_model_str

with open("{0}/dicts/fit_funcs.pkl".format(os.environ["CD_CODE_DIR"]), "rb") as f:
    fit_funcs_dict = pickle.load(f)

NPTS = 100
REG = 1e-8
# NPTS = 1000


def min_coeffs(
    mu,
    omega0,
    Ns,
    model_name,
    H_params,
    symmetries,
    sched,
    ctrls,
    agp_order,
    grid_size,
):
    # NOTE: may want to account for drawing spectral function from
    # smaller system size than current N at some point
    lam_data = np.loadtxt(
        "data_dump/spec_fn_data/{0}_lam_data.txt".format(
            make_model_str(Ns, model_name, H_params, ctrls)
        )
    )
    omega_data = np.loadtxt(
        "data_dump/spec_fn_data/{0}_omega_data.txt".format(
            make_model_str(Ns, model_name, H_params, ctrls)
        )
    )
    spec_fn_grid_data = np.loadtxt(
        "data_dump/spec_fn_data/{0}_spec_fn_grid.txt".format(
            make_model_str(Ns, model_name, H_params, ctrls)
        )
    )

    spec_fn_interp = scipy.interpolate.RectBivariateSpline(
        lam_data, omega_data, spec_fn_grid_data
    )

    tgrid = np.linspace(0, sched.tau, grid_size)
    lamgrid = sched.get_lam(tgrid)

    betas_grid = np.zeros((grid_size, agp_order))

    for j in range(grid_size):
        x = np.linspace(mu, max(omega_data), NPTS)
        y = -1 / x
        fit_func = fit_funcs_dict["bessel"]
        # get weight from spectral function and add regularization, and normalize
        weight = np.sqrt(np.abs(spec_fn_interp(lamgrid[j], x)).flatten())
        weight += REG
        weight /= np.sum(weight)

        # normalize weight function
        weight /= np.sum(weight)
        opt_coeffs, _ = curve_fit(
            fit_func,
            x,
            y,
            p0=np.zeros(agp_order),
            sigma=1 / weight,
        )
        betas_grid[j, :] = opt_coeffs

    # NOTE: may want to put some data saving in here
    return tgrid, betas_grid
