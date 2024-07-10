import os
import sys

import numpy as np

sys.path.append(os.environ["CD_CODE_DIR"])

from cd_protocol import CD_Protocol
from tools.build_ham import build_ham
from tools.lin_alg_calls import calc_fid
from utils.file_naming import make_coeffs_fname, make_evolved_wfs_fname
from utils.grid_utils import get_universal_coeffs_func


def run_time_evolution_universal(
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
    rescale,
    ## not used by all scripts
    save_wf=True,
    coeffs_fname=None,
    wfs_save_append_str=None,
    print_fid=False,
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
        rescale=rescale,
    )
    # add the controls
    ham.init_controls(ctrls, ctrls_couplings, ctrls_args)

    coeffs = np.loadtxt(
        "{0}/coeffs_data/{1}.txt".format(os.environ["CD_CODE_DIR"], coeffs_fname),
        ndmin=1,
    )

    # particularized to alphas
    ham.alphas_interp = get_universal_coeffs_func(coeffs)

    # particularized to chebyshev
    ham.polycoeffs_interp = get_universal_coeffs_func(coeffs)

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

    t_data, wf_data = cd_protocol.matrix_evolve(
        init_state, wfs_fname, save_states=save_wf
    )
    final_state = wf_data[-1, :]
    # print("final state is ", final_state)

    if print_fid:
        print("fidelity is ", calc_fid(targ_state, final_state))
    return final_state
