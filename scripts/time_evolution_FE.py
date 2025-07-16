import os
import sys

import numpy as np

sys.path.append(os.environ["CD_CODE_DIR"])

from cd_protocol import CD_Protocol

from tools.build_ham import build_ham
from tools.lin_alg_calls import calc_fid
from utils.file_IO import open_file, load_data_agp_coeffs, save_data_evolved_wfs
from utils.file_naming import (
    make_data_dump_name,
    make_file_name,
    make_FE_protocol_name,
    make_controls_name,
)
from utils.grid_utils import get_coeffs_interp


def run_time_evolution_FE(
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
    omega=10.0,  # Floquet frequency
    mu=1.0,  # beginning frequency for fitting
    omega0=1.0,  # single-particle reference frequency
    save_protocol_wf=False,
    coeffs_file_name=None,
    coeffs_protocol_name=None,
    coeffs_ctrls_name=None,
    coeffs_sched=None,
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
    )
    # add the controls
    ham.init_controls(ctrls, ctrls_couplings, ctrls_args)

    if agp_order > 0:
        tgrid, betas_grid, _ = load_data_agp_coeffs(
            coeffs_file_name, coeffs_protocol_name, coeffs_ctrls_name
        )
        ham.betas_interp = get_coeffs_interp(coeffs_sched, sched, tgrid, betas_grid)
        ham.omega0 = omega0

    cd_protocol = CD_Protocol(
        ham, AGPtype, ctrls, ctrls_couplings, ctrls_args, sched, grid_size
    )

    save_dirname = "{0}/data_dump".format(os.environ["CD_CODE_DIR"])
    # change to saving for FE
    file_name = make_file_name(
        Ns, model_name, H_params, symmetries, ctrls, boundary_conds
    )
    protocol_name = make_FE_protocol_name(agp_order, 0.0, mu, omega0, grid_size, sched)
    controls_name = make_controls_name(ctrls_couplings, ctrls_args)

    names_list = (file_name, protocol_name, controls_name)

    init_state = ham.get_init_gstate()
    targ_state = ham.get_targ_gstate()

    # change to evolution for FE Hamiltonian
    t_data, wf_data = cd_protocol.matrix_evolve(init_state, omega=omega)
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
