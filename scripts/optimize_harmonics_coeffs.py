import os
import sys

import numpy as np
import scipy

sys.path.append(os.environ["CD_CODE_DIR"])

from cd_protocol import CD_Protocol
from tools.build_ham import build_ham
from tools.calc_coeffs import calc_lanc_coeffs_grid, calc_alphas_grid, calc_gammas_grid
from tools.lin_alg_calls import calc_fid

from utils.file_IO import (
    save_data_agp_coeffs,
    save_data_optimization_fids,
    save_data_evolved_wfs,
)
from utils.file_naming import make_data_dump_name, make_controls_name_no_coeffs
from utils.grid_utils import get_coeffs_interp


def calc_infid(
    coeffs,  # numbers to optimize, array
    harmonics,  # index of harmonics, array
    ham,
    sched,
    ctrls,
    ctrls_couplings,
    agp_order,
    AGPtype,
    norm_type,
    grid_size,
    fname_args_dict,
):
    # set coefficients in given instance
    ctrls_args = []
    for i in range(len(ctrls)):
        ctrls_args.append([sched, *np.append(harmonics, coeffs[i :: len(ctrls)])])
    ham.init_controls(ctrls, ctrls_couplings, ctrls_args)

    # get coefficients
    if agp_order == 0:
        pass
    elif AGPtype == "krylov":
        tgrid, lanc_grid = calc_lanc_coeffs_grid(
            ham,
            grid_size,
            sched,
            agp_order,
            norm_type,
            gs_func=None,
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
        )
        ham.alphas_interp = get_coeffs_interp(sched, sched, tgrid, alphas_grid)

    cd_protocol = CD_Protocol(
        ham, AGPtype, ctrls, ctrls_couplings, ctrls_args, sched, grid_size
    )

    init_state = ham.get_init_gstate()
    targ_state = ham.get_targ_gstate()

    t_data, wf_data = cd_protocol.matrix_evolve(init_state)
    final_wf = wf_data[-1, :]
    fid = calc_fid(targ_state, final_wf)

    # now save relevant data
    fname_args_dict["ctrls_args"] = ctrls_args
    save_dirname = "{0}/data_dump".format(os.environ["CD_CODE_DIR"])
    file_name, protocol_name, _ = make_data_dump_name(*fname_args_dict.values())
    # overwrite last part of names list since don't include coeffs magnitude in fname
    ctrls_name = make_controls_name_no_coeffs(ctrls_couplings, ctrls_args)
    names_list = (file_name, protocol_name, ctrls_name)

    if agp_order > 0:
        save_data_agp_coeffs(*names_list, tgrid, gammas_grid, lanc_grid=lanc_grid)
    save_data_optimization_fids(*names_list, coeffs, fid)
    save_data_evolved_wfs(*names_list, final_wf, tgrid=None, full_wf=None)
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

    fname_args_dict = {
        "Ns": Ns,
        "model_name": model_name,
        "H_params": H_params,
        "symmetries": symmetries,
        "sched": sched,
        "ctrls": ctrls,
        "ctrls_couplings": ctrls_couplings,
        "ctrls_args": None,  # will be replaced when coeffs are filled in
        "agp_order": agp_order,
        "AGPtype": AGPtype,
        "norm_type": norm_type,
        "grid_size": grid_size,
    }

    # do Powell optimization
    bounds = [(-maxfields, maxfields) for _ in range(len(ctrls) * len(ctrls_harmonics))]
    res = scipy.optimize.minimize(
        optim_func,
        args=(
            ctrls_harmonics,
            ham,
            sched,
            ctrls,
            ctrls_couplings,
            agp_order,
            AGPtype,
            norm_type,
            grid_size,
            fname_args_dict,
        ),
        bounds=bounds,
        method="Nelder-Mead",
        options={
            "disp": True,
            "xtol": 1e-4,
            "ftol": 1e-4,
        },
    )


def calc_infid_line(
    scalar,
    harmonics,  # index of harmonics, array
    ham,
    sched,
    ctrls,
    ctrls_couplings,
    agp_order,
    AGPtype,
    norm_type,
    grid_size,
    fname_args_dict,
):
    coeffs = scalar * np.array([1, -1])
    # set coefficients in given instance
    ctrls_args = []
    for i in range(len(ctrls)):
        ctrls_args.append([sched, *np.append(harmonics, coeffs[i :: len(ctrls)])])
    ham.init_controls(ctrls, ctrls_couplings, ctrls_args)

    # get coefficients
    if agp_order == 0:
        pass
    elif AGPtype == "krylov":
        tgrid, lanc_grid = calc_lanc_coeffs_grid(
            ham,
            grid_size,
            sched,
            agp_order,
            norm_type,
            gs_func=None,
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
        )
        ham.alphas_interp = get_coeffs_interp(sched, sched, tgrid, alphas_grid)

    cd_protocol = CD_Protocol(
        ham, AGPtype, ctrls, ctrls_couplings, ctrls_args, sched, grid_size
    )

    init_state = ham.get_init_gstate()
    targ_state = ham.get_targ_gstate()

    t_data, wf_data = cd_protocol.matrix_evolve(init_state)
    final_wf = wf_data[-1, :]
    fid = calc_fid(targ_state, final_wf)

    # now save relevant data
    fname_args_dict["ctrls_args"] = ctrls_args
    save_dirname = "{0}/data_dump".format(os.environ["CD_CODE_DIR"])
    file_name, protocol_name, _ = make_data_dump_name(*fname_args_dict.values())
    # overwrite last part of names list since don't include coeffs magnitude in fname
    ctrls_name = make_controls_name_no_coeffs(ctrls_couplings, ctrls_args)
    names_list = (file_name, protocol_name, ctrls_name)

    if agp_order > 0:
        save_data_agp_coeffs(*names_list, tgrid, gammas_grid, lanc_grid=lanc_grid)
    save_data_optimization_fids(*names_list, coeffs, fid)
    save_data_evolved_wfs(*names_list, final_wf, tgrid=None, full_wf=None)
    print("for controls ", coeffs, " fid is ", fid)
    return 1 - fid


def optim_harmonic_coeffs_line(
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

    optim_func = calc_infid_line

    fname_args_dict = {
        "Ns": Ns,
        "model_name": model_name,
        "H_params": H_params,
        "symmetries": symmetries,
        "sched": sched,
        "ctrls": ctrls,
        "ctrls_couplings": ctrls_couplings,
        "ctrls_args": None,  # will be replaced when coeffs are filled in
        "agp_order": agp_order,
        "AGPtype": AGPtype,
        "norm_type": norm_type,
        "grid_size": grid_size,
    }

    # do Powell optimization
    res = scipy.optimize.minimize_scalar(
        optim_func,
        args=(
            ctrls_harmonics,
            ham,
            sched,
            ctrls,
            ctrls_couplings,
            agp_order,
            AGPtype,
            norm_type,
            grid_size,
            fname_args_dict,
        ),
        bounds=(-3, 3),
        # method="Nelder-Mead",
        # options={
        #     "disp": True,
        #     "xtol": 1e-4,
        #     "ftol": 1e-4,
        # },
    )
