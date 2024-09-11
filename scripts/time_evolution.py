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


def run_time_evolution(
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

    # load relevant coeffs for AGP
    if AGPtype == "commutator" and agp_order > 0:
        tgrid, alphas_grid, _ = load_data_agp_coeffs(
            coeffs_file_name, coeffs_protocol_name, coeffs_ctrls_name
        )
        ham.alphas_interp = get_coeffs_interp(coeffs_sched, sched, tgrid, alphas_grid)
    elif AGPtype == "krylov" and agp_order > 0:
        tgrid, gammas_grid, lgrid = load_data_agp_coeffs(
            coeffs_file_name, coeffs_protocol_name, coeffs_ctrls_name
        )
        ham.lanc_interp = get_coeffs_interp(coeffs_sched, sched, tgrid, lgrid)
        ham.gammas_interp = get_coeffs_interp(coeffs_sched, sched, tgrid, gammas_grid)
    elif agp_order > 0:
        raise ValueError(f"AGPtype {AGPtype} not recognized")

    cd_protocol = CD_Protocol(
        ham, AGPtype, ctrls, ctrls_couplings, ctrls_args, sched, grid_size
    )

    save_dirname = "{0}/data_dump".format(os.environ["CD_CODE_DIR"])
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

    init_state = ham.get_init_gstate()
    targ_state = ham.get_targ_gstate()

    t_data, wf_data = cd_protocol.matrix_evolve(init_state)
    final_state = wf_data[-1, :]

    if save_protocol_wf:
        save_data_evolved_wfs(*names_list, final_state, tgrid=t_data, full_wf=wf_data)
    else:
        save_data_evolved_wfs(*names_list, final_state)

    if print_fid:
        print("fidelity is ", calc_fid(targ_state, final_state))
    return final_state
