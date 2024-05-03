import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy

sys.path.append(os.environ["CD_CODE_DIR"])

from cd_protocol import CD_Protocol
from tools.build_ham import build_ham
from tools.calc_coeffs import calc_alphas_grid, calc_lanc_coeffs_grid, calc_gammas_grid
from tools.lin_alg_calls import calc_fid
from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op
from utils.file_naming import make_base_fname, make_coeffs_fname, make_evolved_wfs_fname
from utils.grid_utils import get_coeffs_interp


def run_iterative_evolution(
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    target_symmetries,
    tau,
    sched,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    agp_order,
    AGPtype,
    norm_type,
    grid_size,
    coeffs_sched,
    coeffs_append_str,
    wfs_append_str,
):
    # load Hamiltonian and initial coefficients from ground state
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
    ham.init_controls(ctrls, ctrls_couplings, ctrls_args)

    coeffs_fname = make_coeffs_fname(
        ham,
        model_name,
        ctrls,
        AGPtype,
        "trace",  # load data from inf T
        grid_size,
        coeffs_sched,  # need separate coeff sched!!
        coeffs_append_str,
    )
    # load relevant coeffs for AGP
    if AGPtype == "commutator":
        tgrid = np.loadtxt("coeffs_data/{0}_alphas_tgrid.txt".format(coeffs_fname))
        alphas_grid = np.loadtxt(
            "coeffs_data/{0}_alphas_grid.txt".format(coeffs_fname), ndmin=2
        )
        ham.alphas_interp = get_coeffs_interp(coeffs_sched, sched, tgrid, alphas_grid)
    elif AGPtype == "krylov":
        tgrid = np.loadtxt("coeffs_data/{0}_lanc_coeffs_tgrid.txt".format(coeffs_fname))
        lgrid = np.loadtxt(
            "coeffs_data/{0}_lanc_coeffs_grid.txt".format(coeffs_fname), ndmin=2
        )
        gammas_grid = np.loadtxt(
            "coeffs_data/{0}_gammas_grid.txt".format(coeffs_fname), ndmin=2
        )
        ham.lanc_interp = get_coeffs_interp(coeffs_sched, sched, tgrid, lgrid)
        ham.gammas_interp = get_coeffs_interp(coeffs_sched, sched, tgrid, gammas_grid)
    else:
        raise ValueError(f"AGPtype {AGPtype} not recognized")

    # loop until fid is converged
    fid = 0
    last_fid = -1
    fids = []

    wfs_fname = None  # unnecessary since not save wfs at each step
    i = 1  # index to track iteration number
    # while abs(fid - last_fid) > 1e-6 and i <= 25:
    while i <= 10:
        cd_protocol = CD_Protocol(
            ham, AGPtype, ctrls, ctrls_couplings, ctrls_args, sched, grid_size
        )
        init_state = ham.get_init_gstate()
        targ_state = ham.get_targ_gstate()
        t_data, wf_data = cd_protocol.matrix_evolve(
            init_state, wfs_fname, save_states=False
        )
        wf_interp = scipy.interpolate.interp1d(
            t_data, wf_data, axis=0, fill_value="extrapolate"
        )
        final_state = wf_data[-1, :]
        last_fid = fid
        fid = calc_fid(targ_state, final_state)
        fids.append(fid)
        print(f"step {i} fidelity is ", fid)
        # print("final state is ", final_state)
        fname = make_coeffs_fname(
            ham,
            model_name,
            ctrls,
            AGPtype,
            norm_type,
            grid_size,
            coeffs_sched,
            "iter_step_{0}".format(i),
        )
        if AGPtype == "commutator":
            tgrid, alpha_grid = calc_alphas_grid(
                ham,
                fname,
                grid_size,
                sched,
                agp_order,
                norm_type,
                gs_func=wf_interp,
                save=True,
            )
            ham.alphas_interp = scipy.interpolate.interp1d(
                tgrid, alpha_grid, axis=0, fill_value="extrapolate"
            )
        elif AGPtype == "krylov":
            lanc_tgrid, lanc_grid = calc_lanc_coeffs_grid(
                ham,
                fname,
                grid_size,
                sched,
                agp_order,
                norm_type,
                gs_func=wf_interp,
                save=True,
            )
            ham.lanc_interp = scipy.interpolate.interp1d(
                lanc_tgrid, lanc_grid, axis=0, fill_value="extrapolate"
            )
            tgrid, gammas_grid = calc_gammas_grid(
                ham,
                fname,
                grid_size,
                sched,
                agp_order,
                norm_type,
                gs_func=wf_interp,
                save=True,
            )
            ham.gammas_interp = scipy.interpolate.interp1d(
                tgrid, gammas_grid, axis=0, fill_value="extrapolate"
            )
        else:
            raise ValueError(f"AGPtype {AGPtype} not recognized")
        i += 1

    wfs_fname = make_evolved_wfs_fname(
        ham,
        model_name,
        ctrls,
        AGPtype,
        norm_type,
        grid_size,
        sched.tau,
        wfs_append_str,
    )
    t_data, wf_data = cd_protocol.matrix_evolve(init_state, wfs_fname, save_states=True)
    final_state = wf_data[-1, :]
    fid = calc_fid(targ_state, final_state)
    fids.append(fid)
    print("fidelity of final iterated state is ", fid)

    fname = make_base_fname(
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
        "iterative",
    )
    return fids, fname


Ns = 8
# model_name = "TFIM_1D"
model_name = "LR_Ising_1D"
# H_params = [1, 1]
H_params = [1, 1, 2]
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

ctrls = []
ctrls_couplings = []
ctrls_args = []

agp_order = 5
AGPtype = "krylov"
# AGPtype = "commutator"
# norm_type = "trace"
norm_type = "ground_state"

grid_size = 1000

coeffs_sched = LinearSchedule(1)  # always use tau = 1 for grid save
coeffs_append_str = "normal"
wfs_append_str = "iterative"

fids, fname = run_iterative_evolution(
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    target_symmetries,
    tau,
    sched,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    agp_order,
    AGPtype,
    norm_type,
    grid_size,
    coeffs_sched,
    coeffs_append_str,
    wfs_append_str,
)

np.savetxt("plots/data/{0}_fids.txt".format(fname), np.array(fids))
