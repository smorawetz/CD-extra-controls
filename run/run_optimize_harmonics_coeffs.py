import os
import sys

import numpy as np
import scipy

sys.path.append(os.environ["CD_CODE_DIR"])

from cd_protocol import CD_Protocol
from tools.build_ham import build_ham
from tools.calc_coeffs import calc_lanc_coeffs_grid, calc_gammas_grid
from tools.lin_alg_calls import calc_fid
from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op
from utils.file_naming import make_base_fname
from utils.grid_utils import get_coeffs_interp


def calc_infid(
    coeffs,  # numbers to optimize, array
    harmonics,  # index of harmonics, array
    ham,
    sched,
    ctrls,
    ctrls_couplings,
    AGPtype,
    grid_size,
    base_fname,
):
    # set coefficients in given instance
    ctrls_args = []
    for i in range(len(ctrls)):
        ctrls_args.append([sched, *np.append(harmonics, coeffs[i :: len(ctrls)])])
    ham.init_controls(ctrls, ctrls_couplings, ctrls_args)

    # get coefficients
    tgrid, lanc_grid = calc_lanc_coeffs_grid(
        ham,
        grid_size,
        sched,
        agp_order,
        norm_type,
        gs_func=None,
        save=False,
    )
    ham.lanc_interp = get_coeffs_interp(sched, sched, tgrid, lanc_grid)  # same sched
    tgrid, gammas_grid = calc_gammas_grid(
        ham,
        grid_size,
        sched,
        agp_order,
        norm_type,
        gs_func=None,
        save=False,
    )
    ham.gammas_interp = get_coeffs_interp(sched, sched, tgrid, gammas_grid)

    cd_protocol = CD_Protocol(
        ham, AGPtype, ctrls, ctrls_couplings, ctrls_args, sched, grid_size
    )

    init_state = ham.get_init_gstate()
    targ_state = ham.get_targ_gstate()

    t_data, wf_data = cd_protocol.matrix_evolve(init_state, None, save_states=False)
    final_state = wf_data[-1, :]
    fid = calc_fid(targ_state, final_state)

    ctrls_write_fname = "optim_ctrls_data/" + base_fname + "_optim_ctrls_coeffs.txt"
    coeffs_grid_fname = "optim_ctrls_data/" + base_fname + "_optim_coeffs_grid.txt"
    tgrid_fname = "optim_ctrls_data/" + base_fname + "_optim_tgrid.txt"
    optim_wf_fname = "optim_ctrls_data/" + base_fname + "_optim_final_wf.txt"

    np.savetxt(optim_wf_fname, final_state)  # save WF, will be optim after final step
    np.savetxt(tgrid_fname, t_data)
    np.savetxt(coeffs_grid_fname, gammas_grid)

    # write to file
    if ctrls_write_fname is not None:
        data_file = open(ctrls_write_fname, "a")
        data_file.write("{0}\t{1}\n".format(coeffs, fid))
        data_file.close()
    print("for controls ", coeffs, " fid is ", fid)
    return 1 - fid


def optim_coeffs(
    ## H params
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    ## schedule params
    tau,
    sched,
    ## controls params
    ctrls,
    ctrls_couplings,
    ctrls_harmonics,
    ## agp params
    agp_order,
    AGPtype,
    norm_type,
    ## simulation params
    grid_size,
    append_str,
    ## optimization params
    maxfields,
):
    ham = build_ham(
        model_name,
        Ns,
        H_params,
        boundary_conds,
        agp_order,
        norm_type,
        sched,
        symmetries=symmetries,
        target_symmetries=symmetries,
    )
    coeffs = np.zeros(len(ctrls) * len(ctrls_harmonics))

    optim_func = calc_infid

    base_fname = make_base_fname(
        Ns,
        model_name,
        H_params,
        symmetries,
        ctrls,
        agp_order,
        AGPtype,
        norm_type,
        grid_size,
        sched,
        append_str,
    )

    # do Powell optimization
    bounds = [(-maxfields, maxfields) for _ in range(len(ctrls))]
    res = scipy.optimize.minimize(
        optim_func,
        coeffs,
        args=(
            ctrls_harmonics,
            ham,
            sched,
            ctrls,
            ctrls_couplings,
            AGPtype,
            grid_size,
            base_fname,
        ),
        bounds=bounds,
        method="Powell",
        options={
            "disp": True,
            "xtol": 1e-4,
            "ftol": 1e-4,
        },
    )


Ns = 4
model_name = "TFIM_1D"
# model_name = "LR_Ising_1D"
H_params = [1, 1]
# H_params = [1, 1, 2]
boundary_conds = "periodic"

symms = ["translation_1d", "spin_inversion"]
symms_args = [[Ns], [Ns]]
symm_nums = [0, 0]
symmetries = {
    symms[i]: (get_symm_op(symms[i], *symms_args[i]), symm_nums[i])
    for i in range(len(symms))
}
target_symmetries = symmetries

tau = 0.01
sched = SmoothSchedule(tau)

ctrls = ["Hc1", "Hc2"]
ctrls_couplings = ["sin", "sin"]
ctrls_harmonics = [1]

agp_order = 1
AGPtype = "krylov"
# AGPtype = "commutator"
norm_type = "trace"
# norm_type = "ground_state"

grid_size = 1000

append_str = "powell"

maxfields = 3

# for Ns in np.arange(5, 10 + 1):
if True:
    symms = ["translation_1d", "spin_inversion"]
    symms_args = [[Ns], [Ns]]
    symm_nums = [0, 0]
    symmetries = {
        symms[i]: (get_symm_op(symms[i], *symms_args[i]), symm_nums[i])
        for i in range(len(symms))
    }
    target_symmetries = symmetries
    optim_coeffs(
        ## H params
        Ns,
        model_name,
        H_params,
        boundary_conds,
        symmetries,
        ## schedule params
        tau,
        sched,
        ## controls params
        ctrls,
        ctrls_couplings,
        ctrls_harmonics,
        ## agp params
        agp_order,
        AGPtype,
        norm_type,
        ## simulation params
        grid_size,
        append_str,
        ## optimization params
        maxfields,
    )
