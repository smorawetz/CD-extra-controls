import os
import sys

import numpy as np
import scipy

sys.path.append(os.environ["CD_CODE_DIR"])

from cd_protocol import CD_Protocol
from tools.build_ham import build_ham
from tools.calc_coeffs import calc_lanc_coeffs_grid, calc_alphas_grid, calc_gammas_grid
from tools.lin_alg_calls import calc_fid
from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op
from utils.file_naming import make_base_fname
from utils.grid_utils import get_coeffs_interp


def calc_infid(
    coeffs,  # numbers to optimize, array
    harmonics,  # index of harmonics, array
    ham,
    sched,
    ctrls,
    ctrls_couplings,
    AGPtype,
    grid_size,
    base_fname,
):
    # set coefficients in given instance
    ctrls_args = []
    for i in range(len(ctrls)):
        ctrls_args.append([sched, *np.append(harmonics, coeffs[i :: len(ctrls)])])
    ham.init_controls(ctrls, ctrls_couplings, ctrls_args)

    # get coefficients
    if AGPtype == "krylov":
        tgrid, lanc_grid = calc_lanc_coeffs_grid(
            ham,
            grid_size,
            sched,
            agp_order,
            norm_type,
            gs_func=None,
            save=False,
        )
        ham.lanc_interp = get_coeffs_interp(
            sched, sched, tgrid, lanc_grid
        )  # same sched
        tgrid, gammas_grid = calc_gammas_grid(
            ham,
            grid_size,
            sched,
            agp_order,
            norm_type,
            gs_func=None,
            save=False,
        )
        ham.gammas_interp = get_coeffs_interp(sched, sched, tgrid, gammas_grid)
    elif AGPtype == "commutator":
        tgrid, alphas_grid = calc_alphas_grid(
            ham,
            grid_size,
            sched,
            agp_order,
            norm_type,
            gs_func=None,
            save=False,
        )
        ham.alphas_interp = get_coeffs_interp(sched, sched, tgrid, alphas_grid)

    cd_protocol = CD_Protocol(
        ham, AGPtype, ctrls, ctrls_couplings, ctrls_args, sched, grid_size
    )

    init_state = ham.get_init_gstate()
    targ_state = ham.get_targ_gstate()

    t_data, wf_data = cd_protocol.matrix_evolve(init_state, None, save_states=False)
    final_state = wf_data[-1, :]
    fid = calc_fid(targ_state, final_state)

    ctrls_write_fname = "optim_ctrls_data/" + base_fname + "_optim_ctrls_coeffs.txt"
    tgrid_fname = "optim_ctrls_data/" + base_fname + "_optim_tgrid.txt"
    np.savetxt(tgrid_fname, tgrid)
    if AGPtype == "krylov":
        lanc_grid_fname = "optim_ctrls_data/" + base_fname + "_optim_lanc_grid.txt"
        gammas_grid_fname = "optim_ctrls_data/" + base_fname + "_optim_gammas_grid.txt"
        np.savetxt(lanc_grid_fname, lanc_grid)
        np.savetxt(gammas_grid_fname, gammas_grid)
    elif AGPtype == "commutator":
        alphas_grid_fname = "optim_ctrls_data/" + base_fname + "_optim_alphas_grid.txt"
        np.savetxt(alphas_grid_fname, alphas_grid)
    optim_wf_fname = "optim_ctrls_data/" + base_fname + "_optim_final_wf.txt"

    # write to file
    if ctrls_write_fname is not None:
        data_file = open(ctrls_write_fname, "a")
        data_file.write("{0}\t{1}\n".format(coeffs, fid))
        data_file.close()
    print("for controls ", coeffs, " fid is ", fid)
    return 1 - fid


def optim_harmonic_coeffs(
    ## used for all scripts
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
    ctrls_harmonics,
    agp_order,
    AGPtype,
    norm_type,
    grid_size,
    ## not used for all scripts
    append_str=None,
    maxfields=None,
):
    ham = build_ham(
        model_name,
        Ns,
        H_params,
        boundary_conds,
        model_kwargs,
        agp_order,
        norm_type,
        sched,
        symmetries=symmetries,
        target_symmetries=target_symmetries,
    )
    coeffs = np.zeros(len(ctrls) * len(ctrls_harmonics))

    optim_func = calc_infid

    base_fname = make_base_fname(
        Ns,
        model_name,
        H_params,
        symmetries,
        ctrls,
        agp_order,
        AGPtype,
        norm_type,
        grid_size,
        sched,
        append_str,
    )

    # do Powell optimization
    bounds = [(-maxfields, maxfields) for _ in range(len(ctrls))]
    res = scipy.optimize.minimize(
        optim_func,
        coeffs,
        args=(
            ctrls_harmonics,
            ham,
            sched,
            ctrls,
            ctrls_couplings,
            AGPtype,
            grid_size,
            base_fname,
        ),
        bounds=bounds,
        method="Powell",
        options={
            "disp": True,
            "xtol": 1e-4,
            "ftol": 1e-4,
        },
    )
