import os
import sys

import numpy as np
import scipy

sys.path.append(os.environ["CD_CODE_DIR"])

from cd_protocol import CD_Protocol
from tools.build_ham import build_ham
from tools.calc_coeffs import calc_alphas_grid, calc_lanc_coeffs_grid, calc_gammas_grid
from tools.lin_alg_calls import calc_fid
from utils.file_IO import load_data_agp_coeffs, save_data_evolved_wfs
from utils.file_naming import (
    make_data_dump_name,
    make_file_name,
    make_protocol_name,
    make_controls_name,
)
from utils.grid_utils import get_coeffs_interp


def do_iterative_evolution(
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
    ## not used by all scripts
    save_protocol_wf=False,
    coeffs_file_name=None,
    coeffs_protocol_name=None,
    coeffs_ctrls_name=None,
    coeffs_sched=None,
    print_fid=False,
):
    # load Hamiltonian and initial coefficients from ground state
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
    ham.init_controls(ctrls, ctrls_couplings, ctrls_args)

    # load relevant coeffs for AGP
    if AGPtype == "commutator":
        tgrid, alphas_grid, _ = load_data_agp_coeffs(
            coeffs_file_name, coeffs_protocol_fname, coeffs_ctrls_fname
        )
        ham.alphas_interp = get_coeffs_interp(coeffs_sched, sched, tgrid, alphas_grid)
    elif AGPtype == "krylov":
        tgrid, gammas_grid, lgrid = load_data_agp_coeffs(
            coeffs_file_name, coeffs_protocol_fname, coeffs_ctrls_fname
        )
        ham.lanc_interp = get_coeffs_interp(coeffs_sched, sched, tgrid, lgrid)
        ham.gammas_interp = get_coeffs_interp(coeffs_sched, sched, tgrid, gammas_grid)
    else:
        raise ValueError(f"AGPtype {AGPtype} not recognized")

    # loop until fid is converged
    fid = 0
    last_fid = -1
    fids = []

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

    i = 1  # index to track iteration number
    while i <= 10:
        cd_protocol = CD_Protocol(
            ham, AGPtype, ctrls, ctrls_couplings, ctrls_args, sched, grid_size
        )
        init_state = ham.get_init_gstate()
        targ_state = ham.get_targ_gstate()
        t_data, wf_data = cd_protocol.matrix_evolve(
            init_state, wfs_fname, save_states=False
        )
        wf_interp = scipy.interpolate.interp1d(
            t_data, wf_data, axis=0, fill_value="extrapolate"
        )
        final_state = wf_data[-1, :]
        last_fid = fid
        fid = calc_fid(targ_state, final_state)
        fids.append(fid)
        if print_fid:
            print(f"step {i} fidelity is ", fid)

        if AGPtype == "commutator":
            tgrid, alpha_grid = calc_alphas_grid(
                ham,
                grid_size,
                sched,
                agp_order,
                norm_type,
                gs_func=wf_interp,
            )
            ham.alphas_interp = scipy.interpolate.interp1d(
                tgrid, alpha_grid, axis=0, fill_value="extrapolate"
            )
            save_data_agp_coeffs(
                data_dump_fname + f"_iter_step{i}", tgrid, alphas_grid, lanc_grid=None
            )
        elif AGPtype == "krylov":
            lanc_tgrid, lanc_grid = calc_lanc_coeffs_grid(
                ham,
                grid_size,
                sched,
                agp_order,
                norm_type,
                gs_func=wf_interp,
            )
            ham.lanc_interp = scipy.interpolate.interp1d(
                lanc_tgrid, lanc_grid, axis=0, fill_value="extrapolate"
            )
            tgrid, gammas_grid = calc_gammas_grid(
                ham,
                grid_size,
                sched,
                agp_order,
                norm_type,
                gs_func=wf_interp,
            )
            ham.gammas_interp = scipy.interpolate.interp1d(
                tgrid, gammas_grid, axis=0, fill_value="extrapolate"
            )
            save_data_agp_coeffs(
                data_dump_fname, tgrid, gammas_grid, lanc_grid=lanc_grid
            )
        else:
            raise ValueError(f"AGPtype {AGPtype} not recognized")
        i += 1

    t_data, wf_data = cd_protocol.matrix_evolve(init_state, wfs_fname, save_states=True)
    final_state = wf_data[-1, :]
    fid = calc_fid(targ_state, final_state)
    fids.append(fid)
    if print_fid:
        print("fidelity of final iterated state is ", fid)

    if save_protocol_wf:
        save_data_evolved_wfs(*names_list, final_state, tgrid=t_data, full_wf=wf_data)
    else:
        save_data_evolved_wfs(*names_list, final_state)
    return fids, fname
