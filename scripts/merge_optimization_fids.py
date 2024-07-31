import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import glob
import h5py
import numpy as np

from utils.file_IO import merge_data_optimization_fids, load_raw_data_optimization_fids
from utils.file_naming import make_data_dump_name, make_controls_name_no_coeffs


def run_merge(
    Ns,
    model_name,
    H_params,
    symmetries,
    ## schedule params
    sched,
    ## controls params
    ctrls,
    ctrls_couplings,
    ctrls_args,
    ## agp params
    agp_order,
    AGPtype,
    norm_type,
    ## simulation params
    grid_size,
):
    file_name, protocol_name, _ = make_data_dump_name(
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
    ctrls_name = make_controls_name_no_coeffs(ctrls_couplings, ctrls_args)
    names_list = (file_name, protocol_name, ctrls_name)
    coeffs, fids = load_raw_data_optimization_fids(*names_list)
    merge_data_optimization_fids(*names_list, coeffs, fids)
