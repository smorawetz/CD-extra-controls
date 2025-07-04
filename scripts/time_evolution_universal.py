import os
import sys

import numpy as np

sys.path.append(os.environ["CD_CODE_DIR"])

from cd_protocol import CD_Protocol
from tools.build_ham import build_ham
from tools.calc_universal_fit_coeffs import fit_universal_coeffs
from tools.lin_alg_calls import calc_fid
from utils.file_IO import save_data_evolved_wfs
from utils.file_naming import (
    make_fit_coeffs_fname,
    make_file_name,
    make_universal_protocol_name,
    make_controls_name,
)
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
    ## not used by all scripts
    rescale=1,
    window_start=0.5,
    window_end=1.0,
    save_protocol_wf=False,
    print_fid=False,
    print_states=False,
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

    coeffs = fit_universal_coeffs(agp_order, AGPtype, window_start, window_end)

    if "commutator" in AGPtype and agp_order > 0:
        ham.alphas_interp = get_universal_coeffs_func(coeffs)
    elif "chebyshev" in AGPtype and agp_order > 0:
        ham.polycoeffs_interp = get_universal_coeffs_func(coeffs)

    cd_protocol = CD_Protocol(
        ham, AGPtype, ctrls, ctrls_couplings, ctrls_args, sched, grid_size
    )

    save_dirname = "{0}/data_dump".format(os.environ["CD_CODE_DIR"])

    file_name = make_file_name(
        Ns, model_name, H_params, symmetries, ctrls, boundary_conds
    )
    protocol_name = make_universal_protocol_name(
        AGPtype, norm_type, agp_order, window_start, window_end, grid_size, sched
    )
    controls_name = make_controls_name(ctrls_couplings, ctrls_args)
    names_list = [file_name, protocol_name, controls_name]

    init_state = ham.get_init_gstate()
    targ_state = ham.get_targ_gstate()

    t_data, wf_data = cd_protocol.matrix_evolve(init_state)
    final_state = wf_data[-1, :]

    if save_protocol_wf:
        save_data_evolved_wfs(*names_list, final_state, tgrid=t_data, full_wf=wf_data)
    else:
        save_data_evolved_wfs(*names_list, final_state)

    if print_fid:
        print("Log fidelity:", np.log(calc_fid(targ_state, final_state)))
    if print_states:
        print("init state is\n", init_state)
        print("final state is\n", final_state)
        print("target state is\n", targ_state)
    return final_state, targ_state
