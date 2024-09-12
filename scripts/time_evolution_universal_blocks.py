import os
import sys

import numpy as np

sys.path.append(os.environ["CD_CODE_DIR"])

from cd_protocol import CD_Protocol
from tools.build_ham import build_ham
from tools.calc_universal_fit_coeffs import fit_universal_coeffs
from tools.lin_alg_calls import calc_fid
from utils.file_IO import save_data_evolved_wfs_blocks
from utils.file_naming import (
    make_fit_coeffs_fname,
    make_file_name,
    make_universal_protocol_name,
    make_controls_name,
)
from utils.grid_utils import get_universal_coeffs_func


def run_time_evolution_universal_blocks(
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
    rescale=1,
    window_start=0.5,
    window_end=1.0,
    save_protocol_wf=False,
    print_fid=False,
    print_states=False,
):
    coeffs = fit_universal_coeffs(agp_order, AGPtype, window_start, window_end)
    coeffs_fname = make_fit_coeffs_fname(AGPtype, agp_order, window_start, window_end)
    np.savetxt(
        "{0}/universal_coeffs/{1}.txt".format(os.environ["CD_CODE_DIR"], coeffs_fname),
        coeffs,
    )
    target_wfs_dict = {}  # populate with k: targ_wf
    final_wfs_dict = {}  # populate with k: final_wf
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
            rescale=rescale,
        )
        # add the controls
        ham.init_controls(ctrls, ctrls_couplings, ctrls_args)

        if agp_order > 0:
            coeffs_fname = make_fit_coeffs_fname(
                AGPtype, agp_order, window_start, window_end
            )
            coeffs = np.loadtxt(
                "{0}/universal_coeffs/{1}.txt".format(
                    os.environ["CD_CODE_DIR"], coeffs_fname
                ),
                ndmin=1,
            )

        # universal can only be commutator (straight polynomial) or chebyshev
        if AGPtype == "commutator" and agp_order > 0:
            ham.alphas_interp = get_universal_coeffs_func(coeffs)
        elif AGPtype == "chebyshev" and agp_order > 0:
            ham.polycoeffs_interp = get_universal_coeffs_func(coeffs)

        cd_protocol = CD_Protocol(
            ham, AGPtype, ctrls, ctrls_couplings, ctrls_args, sched, grid_size
        )

        init_state = ham.get_init_gstate()
        targ_state = ham.get_targ_gstate()

        t_data, wf_data = cd_protocol.matrix_evolve(init_state)
        final_state = wf_data[-1, :]

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

    file_name = make_file_name(full_Ns, model_name, H_params, symmetries, ctrls)
    protocol_name = make_universal_protocol_name(
        AGPtype, norm_type, agp_order, window_start, window_end, grid_size, sched
    )
    controls_name = make_controls_name(ctrls_couplings, ctrls_args)
    names_list = [file_name, protocol_name, controls_name]

    save_data_evolved_wfs_blocks(*names_list, final_wf)

    return final_wf, target_wf
