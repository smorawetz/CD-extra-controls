import os
import sys

import numpy as np

sys.path.append(os.environ["CD_CODE_DIR"])

from cd_hamiltonian import Hamiltonian_CD
from tools.build_ham import build_ham
from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op
from utils.file_naming import make_coeffs_fname

EDGE_OFFSET_FACTOR = 1000


def mod_tgrid(t_grid):
    """Modify the time-grid by shifting the end points so that
    any singularity of the matrix is eliminated by perturbing
    away from special points
    Parameters:
        t_grid (np.array):      Time grid to modify
    """
    t_grid[0] += (t_grid[1] - t_grid[0]) / EDGE_OFFSET_FACTOR
    t_grid[-1] -= (t_grid[-1] - t_grid[-2]) / EDGE_OFFSET_FACTOR
    return t_grid


## want to save coeffs with standardized grid name, can use to
## then load later and do evolution with easily. probably
## use linear schedule, and do once with a larger grid size
## which should hopefully avoid any problems
def calc_alphas_grid(
    ham, fname, grid_size, sched, agp_order, norm_type, gs_func=None, save=True
):
    """Compute the coefficients for the AGP in the commutator ansatz
    on a grid covering the whole protocol, and save them to a file.
    This can later be indexed and translated for repeated use in
    evolution type problems
    Parameters:
        ham (Hamiltonian_CD):       Counterdiabatic Hamiltonian of interest
        fname (str):                Name of file to store in
        grid_size (int):            Number of time steps to take
        sched (Schedule):           Schedule object that encodes $\lambda(t)$
        agp_order (int):            Order of the AGP to compute
        norm_type (str):            Either "trace" or "ground_state" for the norm
        gs_func (function):         Function to compute the ground state of the Hamiltonian
        save (bool):                Whether to save the coefficients to a file
    """
    lam_grid = np.linspace(0, 1, grid_size)
    t_grid = mod_tgrid(sched.get_t(lam_grid))
    alphas_grid = np.zeros((grid_size, agp_order))
    for i in range(grid_size):
        gstate = gs_func(t_grid[i]) if gs_func is not None else None
        alphas_grid[i, :] = ham.calc_alphas(t_grid[i], norm_type, gstate=gstate)
    if save:
        np.savetxt("coeffs_data/{0}_alphas_tgrid.txt".format(fname), t_grid)
        np.savetxt("coeffs_data/{0}_alphas_grid.txt".format(fname), alphas_grid)
    return t_grid, alphas_grid


def calc_lanc_coeffs_grid(
    ham, fname, grid_size, sched, agp_order, norm_type, gs_func=None, save=True
):
    """Compute the Lanczos coefficients on a grid covering the whole protocol,
    and save them to a file.  This can later be indexed and translated for repeated use
    in e.g. evolution
    Parameters:
        ham (Hamiltonian_CD):       Counterdiabatic Hamiltonian of interest
        fname (str):                Name of file to store in
        grid_size (int):            Number of time steps to take
        sched (Schedule):           Schedule object that encodes $\lambda(t)$
        agp_order (int):            Order of the AGP to compute
        norm_type (str):            Either "trace" or "ground_state" for the norm
        gs_func (function):         Function to compute the ground state of the Hamiltonian
        save (bool):                Whether to save the coefficients to a file
    """
    lam_grid = np.linspace(0, 1, grid_size)
    t_grid = mod_tgrid(sched.get_t(lam_grid))
    lanc_grid = np.zeros((grid_size, 2 * agp_order + 1))
    for i in range(grid_size):
        gstate = gs_func(t_grid[i]) if gs_func is not None else None
        lanc_grid[i, :] = ham.calc_lanc_coeffs(t_grid[i], norm_type, gstate=gstate)
    if save:
        np.savetxt("coeffs_data/{0}_lanc_coeffs_tgrid.txt".format(fname), t_grid)
        np.savetxt("coeffs_data/{0}_lanc_coeffs_grid.txt".format(fname), lanc_grid)
    return t_grid, lanc_grid


def calc_gammas_grid(
    ham, fname, grid_size, sched, agp_order, norm_type, gs_func=None, save=True
):
    """Compute the coefficients for the AGP in the Krylov space ansatz
    on a grid covering the whole protocol, and save them to a file.
    This can later be indexed and translated for repeated use in
    evolution type problems
    Parameters:
        ham (Hamiltonian_CD):       Counterdiabatic Hamiltonian of interest
        fname (str):                Name of file to store in
        grid_size (int):            Number of time steps to take
        sched (Schedule):           Schedule object that encodes $\lambda(t)$
        agp_order (int):            Order of the AGP to compute
        norm_type (str):            Either "trace" or "ground_state" for the norm
        gs_func (function):         Function to compute the ground state of the Hamiltonian
        save (bool):                Whether to save the coefficients to a file
    """
    lam_grid = np.linspace(0, 1, grid_size)
    t_grid = mod_tgrid(sched.get_t(lam_grid))
    gammas_grid = np.zeros((grid_size, agp_order))
    for i in range(grid_size):
        gammas_grid[i, :] = ham.calc_gammas(t_grid[i])
    if save:
        np.savetxt("coeffs_data/{0}_gammas_tgrid.txt".format(fname), t_grid)
        np.savetxt("coeffs_data/{0}_gammas_grid.txt".format(fname), gammas_grid)
    return t_grid, gammas_grid
