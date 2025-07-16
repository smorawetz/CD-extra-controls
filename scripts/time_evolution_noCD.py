import os
import sys

import numpy as np

sys.path.append(os.environ["CD_CODE_DIR"])

from cd_protocol import CD_Protocol
from tools.build_ham import build_ham
from tools.lin_alg_calls import calc_fid
from utils.file_IO import open_file, load_data_agp_coeffs, save_data_evolved_wfs
from utils.file_naming import make_data_dump_name
from utils.grid_utils import get_coeffs_interp

# can use QuSpin builtin methods since have no need to construct commutators


def run_time_evolution_noCD(
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

    save_dirname = "{0}/data_dump".format(os.environ["CD_CODE_DIR"])
    names_list = make_data_dump_name(
        Ns,
        model_name,
        H_params,
        boundary_conds,
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

    init_state = ham.get_init_gstate()
    targ_state = ham.get_targ_gstate()

    t_data = np.linspace(0, tau, grid_size)
    wf_data = ham.bareH.evolve(init_state, t_data[0], t_data)

    final_state = wf_data[:, -1]

    if save_protocol_wf:
        save_data_evolved_wfs(*names_list, final_state, tgrid=t_data, full_wf=wf_data)
    else:
        save_data_evolved_wfs(*names_list, final_state)

    if print_fid:
        print("Fidelity:", calc_fid(targ_state, final_state))
        print("Log fidelity:", np.log(calc_fid(targ_state, final_state)))
    if print_states:
        print("init state is\n", init_state)
        print("final state is\n", final_state)
        print("target state is\n", targ_state)
    return final_state, targ_state
