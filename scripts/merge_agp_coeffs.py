import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import glob
import h5py
import numpy as np

from utils.file_IO import merge_data_agp_coeffs, load_raw_data_agp_coeffs
from utils.file_naming import make_data_dump_name, combine_names


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
    tgrid, coeff_grid, lanc_grid = load_raw_data_agp_coeffs(*names_list)
    merge_data_agp_coeffs(*names_list, tgrid, coeff_grid, lanc_grid=lanc_grid)
