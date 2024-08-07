import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import glob
import h5py
import numpy as np

from utils.file_IO import (
    merge_data_agp_coeffs,
    load_raw_data_agp_coeffs,
    merge_data_evolved_wfs,
    load_raw_data_evolved_wfs,
    merge_data_optimization_fids,
    load_raw_data_optimization_fids,
)
from utils.file_naming import (
    make_data_dump_name,
    make_universal_protocol_name,
    combine_names,
)


def run_agp_coeffs_merge(
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


def run_evolved_wfs_merge(
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


def run_universal_evolved_wfs_merge(
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
    window_start,
    window_end,
    ## simulation params
    grid_size,
):
    file_name, _, controls_name = make_data_dump_name(
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
    protocol_name = make_universal_protocol_name(
        AGPtype, norm_type, agp_order, window_start, window_end, grid_size, sched
    )
    names_list = [file_name, protocol_name, controls_name]
    final_wf, tgrid, full_wf = load_raw_data_evolved_wfs(*names_list)
    merge_data_evolved_wfs(*names_list, final_wf, tgrid, full_wf)


def run_optimization_fids_merge(
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
