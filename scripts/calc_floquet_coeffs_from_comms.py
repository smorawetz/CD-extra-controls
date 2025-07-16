import os
import sys

import numpy as np
import scipy

sys.path.append(os.environ["CD_CODE_DIR"])

from tools.build_ham import build_ham
from tools.bessel_to_monomials import buildmat
from tools.calc_coeffs import calc_alphas_grid
from utils.file_IO import save_data_agp_coeffs
from utils.file_naming import make_file_name, make_FE_protocol_name, make_controls_name


def get_floquet_coeffs_from_comm_coeffs(
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
    # used by this script in particular
    omega0=1.0,
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
    tgrid, alphas_grid = calc_alphas_grid(
        ham,
        grid_size,
        sched,
        agp_order,
        norm_type,
        # TODO: fix this to work if there are extra controls
        gs_func=ham.get_bare_inst_gstate,
    )

    mat = buildmat(agp_order, omega0)
    betas_grid = np.einsum("ij,kj->ki", mat, alphas_grid)

    file_name = make_file_name(
        Ns, model_name, H_params, symmetries, ctrls, boundary_conds
    )
    protocol_name = make_FE_protocol_name(
        agp_order,
        0.0,  # this omega does not change coefficients so just pass None
        omega0,
        grid_size,
        sched,
    )
    controls_name = make_controls_name(ctrls_couplings, ctrls_args)
    save_data_agp_coeffs(file_name, protocol_name, controls_name, tgrid, betas_grid)
    return tgrid, betas_grid
