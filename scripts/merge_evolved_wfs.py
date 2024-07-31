import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import glob
import h5py
import numpy as np

from utils.file_IO import merge_data_evolved_wfs, load_raw_data_evolved_wfs
from utils.file_naming import (
    make_data_dump_name,
    make_file_name,
    make_protocol_name,
    make_controls_name,
)


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
    final_wf, tgrid, full_wf = load_raw_data_evolved_wfs(*names_list)
    merge_data_evolved_wfs(*names_list, final_wf, tgrid, full_wf)
