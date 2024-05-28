import os
import sys

import numpy as np

sys.path.append(os.environ["CD_CODE_DIR"])

from cd_protocol import CD_Protocol
from tools.build_ham import build_ham
from tools.lin_alg_calls import calc_fid
from utils.file_naming import make_coeffs_fname, make_evolved_wfs_fname
from utils.grid_utils import get_coeffs_interp


def run_time_evolution(
    ## used by all scripts
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    target_symmetries,
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
    coeffs_fname=None,
    coeffs_sched=None,
    wfs_save_append_str=None,
):
    ham = build_ham(
        model_name,
        Ns,
        H_params,
        boundary_conds,
        agp_order,
        norm_type,
        sched,
        symmetries=symmetries,
        target_symmetries=symmetries,
    )
    # add the controls
    ham.init_controls(ctrls, ctrls_couplings, ctrls_args)

    # load relevant coeffs for AGP
    fname = "{0}/coeffs_data/{1}".format(os.environ["CD_CODE_DIR"], coeffs_fname)
    if AGPtype == "commutator":
        tgrid = np.loadtxt("{0}_alphas_tgrid.txt".format(fname))
        alphas_grid = np.loadtxt("{0}_alphas_grid.txt".format(fname), ndmin=2)
        ham.alphas_interp = get_coeffs_interp(coeffs_sched, sched, tgrid, alphas_grid)
    elif AGPtype == "krylov":
        tgrid = np.loadtxt("{0}_lanc_coeffs_tgrid.txt".format(fname))
        lgrid = np.loadtxt("{0}_lanc_coeffs_grid.txt".format(fname), ndmin=2)
        gammas_grid = np.loadtxt("{0}_gammas_grid.txt".format(fname), ndmin=2)
        ham.lanc_interp = get_coeffs_interp(coeffs_sched, sched, tgrid, lgrid)
        ham.gammas_interp = get_coeffs_interp(coeffs_sched, sched, tgrid, gammas_grid)
    else:
        raise ValueError(f"AGPtype {AGPtype} not recognized")

    cd_protocol = CD_Protocol(
        ham, AGPtype, ctrls, ctrls_couplings, ctrls_args, sched, grid_size
    )

    init_state = ham.get_init_gstate()

    wfs_fname = make_evolved_wfs_fname(
        ham,
        model_name,
        ctrls,
        AGPtype,
        norm_type,
        grid_size,
        sched.tau,
        wfs_save_append_str,
    )

    init_state = ham.get_init_gstate()

    targ_state = ham.get_targ_gstate()
    # print("targ state is ", targ_state)

    t_data, wf_data = cd_protocol.matrix_evolve(init_state, wfs_fname, save_states=True)
    final_state = wf_data[-1, :]
    # print("final state is ", final_state)

    print("fidelity is ", calc_fid(targ_state, final_state))
    return t_data, final_state
