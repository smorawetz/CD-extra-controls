import os
import sys

import numpy as np

sys.path.append(os.environ["CD_CODE_DIR"])

from cd_protocol import CD_Protocol
from tools.build_ham import build_ham
from tools.lin_alg_calls import calc_fid
from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op
from utils.file_naming import make_coeffs_fname, make_evolved_wfs_fname
from utils.grid_utils import get_coeffs_interp

model_name = "TFIM_1D"
Ns = 16
H_params = [1, 1]
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
couplings = []
couplings_args = []

agp_order = 1
AGPtype = "krylov"
# AGPtype = "commutator"
norm_type = "trace"

grid_size = 1000

coeffs_sched = LinearSchedule(1)  # always use tau = 1 for grid save
coeffs_append_str = "no_ctrls"
wfs_append_str = "no_ctrls"

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
coeffs_fname = make_coeffs_fname(
    ham,
    model_name,
    ctrls,
    AGPtype,
    norm_type,
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

cd_protocol = CD_Protocol(
    ham, AGPtype, ctrls, couplings, couplings_args, sched, grid_size
)

init_state = ham.get_init_gstate()

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

init_state = ham.get_init_gstate()

targ_state = ham.get_targ_gstate()
# print("targ state is ", targ_state)

final_state = cd_protocol.matrix_evolve(init_state, wfs_fname, save_states=True)
# print("final state is ", final_state)

print("fidelity is ", calc_fid(targ_state, final_state))
