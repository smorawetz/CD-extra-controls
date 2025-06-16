import os
import sys

import numpy as np

sys.path.append(os.environ["CD_CODE_DIR"])

from cd_protocol import CD_Protocol
from tools.build_ham import build_ham
from tools.calc_universal_fit_coeffs import fit_universal_coeffs
from tools.lin_alg_calls import calc_fid
from utils.file_IO import load_data_agp_coeffs, save_data_evolved_wfs_blocks
from utils.file_naming import make_data_dump_name
from utils.grid_utils import get_coeffs_interp


def run_time_evolution_blocks(
    ## used by all scripts
    full_Ns,
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
    kblocks=[],
    block_Ns=[2],
    save_protocol_wf=False,
    coeffs_file_name=None,
    coeffs_protocol_name=None,
    coeffs_ctrls_name=None,
    coeffs_sched=None,
    print_fid=True,
):
    # replace thing for loading coefficients
    target_wfs_dict = {}  # populate with k: targ_wf
    final_wfs_dict = {}  # populate with k: final_wf
    fid = 1
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
        # add the controls
        ham.init_controls(ctrls, ctrls_couplings, ctrls_args)

        if "commutator" in AGPtype and agp_order > 0:
            tgrid, alphas_grid, _ = load_data_agp_coeffs(
                coeffs_file_name, coeffs_protocol_name, coeffs_ctrls_name
            )
            ham.alphas_interp = get_coeffs_interp(
                coeffs_sched, sched, tgrid, alphas_grid
            )
        else:
            raise ValueError(
                "AGPtype {0} is not supported for blocks evolution".format(AGPtype)
            )

        cd_protocol = CD_Protocol(
            ham, AGPtype, ctrls, ctrls_couplings, ctrls_args, sched, grid_size
        )

        init_state = ham.get_init_gstate()
        targ_state = ham.get_targ_gstate()

        t_data, wf_data = cd_protocol.matrix_evolve(init_state)
        final_state = wf_data[-1, :]

        if print_fid:
            fid *= calc_fid(final_state, targ_state)

        target_wfs_dict[k] = targ_state
        final_wfs_dict[k] = final_state

    block_wf_len = max([len(wf) for wf in final_wfs_dict.values()])
    final_wf = np.zeros((len(kblocks), 1 + block_wf_len), dtype=np.complex128)
    target_wf = np.zeros((len(kblocks), 1 + block_wf_len), dtype=np.complex128)
    for n in range(len(kblocks)):
        k = kblocks[n]
        final_wf[n, 0] = k
        final_wf[n, 1 : 1 + len(final_wfs_dict[k])] = final_wfs_dict[k]
        target_wf[n, 0] = k
        target_wf[n, 1 : 1 + len(target_wfs_dict[k])] = target_wfs_dict[k]

    save_dirname = "{0}/data_dump".format(os.environ["CD_CODE_DIR"])
    names_list = make_data_dump_name(
        full_Ns,
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

    save_data_evolved_wfs_blocks(*names_list, final_wf)

    if print_fid:
        print("Log fidelity: {0}".format(np.log(fid)))

    return final_wf, target_wf
