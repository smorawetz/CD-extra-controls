import os
import sys

import numpy as np
import scipy

sys.path.append(os.environ["CD_CODE_DIR"])

from ham_controls.build_controls_ham import get_H_controls_gs
from tools.build_ham import build_ham
from tools.calc_coeffs import calc_lanc_coeffs_grid, calc_gammas_grid
from utils.file_IO import save_data_agp_coeffs
from utils.file_naming import make_data_dump_name


def calc_kry_coeffs(
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
        symmetries,
        target_symmetries,
    )
    ham.init_controls(ctrls, ctrls_couplings, ctrls_args)

    # now call function to compute alphas
    lanc_tgrid, lanc_grid = calc_lanc_coeffs_grid(
        ham,
        grid_size,
        sched,
        agp_order,
        norm_type,
        # TODO: fix this to work if there eare extra controls
        gs_func=ham.get_bare_inst_gstate,
    )
    ham.lanc_interp = scipy.interpolate.interp1d(lanc_tgrid, lanc_grid, axis=0)
    tgrid, gammas_grid = calc_gammas_grid(
        ham,
        grid_size,
        sched,
        agp_order,
        norm_type,
        # TODO: fix this to work if there are extra controls
        gs_func=ham.get_bare_inst_gstate,
    )

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
    save_data_agp_coeffs(*names_list, tgrid, gammas_grid, lanc_grid=lanc_grid)
    return tgrid, lanc_grid, gammas_grid
