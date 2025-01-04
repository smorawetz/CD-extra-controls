import os
import sys

import numpy as np

sys.path.append(os.environ["CD_CODE_DIR"])

from agp.commutator_ansatz import (
    compute_Rs,
    combine_Rs_blocks,
    inf_temp_R,
    zero_temp_R,
    compute_commutators,
)
from tools.build_ham import build_ham
from utils.file_IO import save_data_agp_coeffs
from utils.file_naming import make_data_dump_name


EDGE_OFFSET_FACTOR = 10


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


def calc_comm_coeffs_blocks(
    ## used by all scripts
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    target_symmetries,
    model_kwargs,
    tau,
    sched,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    agp_order,
    AGPtype,
    norm_type,
    grid_size,
    # not used by all scripts
    kblocks=[],
    block_Ns=[2],
):

    lam_grid = np.linspace(0, 1, grid_size)
    t_grid = mod_tgrid(sched.get_t(lam_grid))

    if norm_type == "trace":
        calc_R = inf_temp_R
        args = []
    elif norm_type == "ground_state":
        calc_R = zero_temp_R
        args = [gstate]

    Rvecs_list = []
    Rmats_list = []

    for k in kblocks:
        H_params[-1] = k
        ham = build_ham(
            model_name,
            block_Ns,
            H_params,
            boundary_conds,
            model_kwargs,
            agp_order,
            norm_type,
            sched,
            symmetries=symmetries,
            target_symmetries=target_symmetries,
        )
        ham.init_controls(ctrls, ctrls_couplings, ctrls_args)

        Rvecs_grid = np.zeros((grid_size, agp_order))
        Rmats_grid = np.zeros((grid_size, agp_order, agp_order))

        for i in range(grid_size):
            t = t_grid[i]
            H = ham.bareH.tocsr(time=t) if ham.sparse else ham.bareH.toarray(time=t)
            dlamH = ham.dlamH.tocsr(time=t) if ham.sparse else ham.dlamH.toarray(time=t)
            commutators = compute_commutators(agp_order, H, dlamH)
            Rmat, Rvec = compute_Rs(agp_order, calc_R, commutators, args)
            Rmats_grid[i, :, :] = Rmat
            Rvecs_grid[i, :] = Rvec

        Rvecs_list.append(Rvecs_grid)
        Rmats_list.append(Rmats_grid)

    Rmat_combined, Rvec_combined = combine_Rs_blocks(Rmats_list, Rvecs_list)

    # now loop over grid which first index indexes the time, and compute alphas
    alphas_grid = np.zeros((grid_size, agp_order))
    for i in range(grid_size):
        alphas_grid[i, :] = np.linalg.solve(
            Rmat_combined[i, :, :], -Rvec_combined[i, :]
        )

    H_params[-1] = "all"  # save under all so know it's for all blocks
    names_list = make_data_dump_name(
        Ns,
        model_name,
        H_params,
        symmetries,
        sched,
        ctrls,
        ctrls_couplings,
        ctrls_args,
        agp_order,
        AGPtype,
        norm_type,
        grid_size,
    )
    save_data_agp_coeffs(*names_list, t_grid, alphas_grid)

    return t_grid, alphas_grid
