import os
import sys

import numpy as np

sys.path.append(os.environ["CD_CODE_DIR"])

from tools.schedules import LinearSchedule
from tools.build_ham import build_ham
from tools.symmetries import get_symm_op
from tools.calc_coeffs import save_lanc_coeffs
from tools.calc_coeffs import save_alphas
from utils.file_naming import make_coeffs_fname

# things to run here
model_name = "TFIM_1D"
Ns = 10
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

tau = 1
sched = LinearSchedule(tau)
ctrls = []

agp_order = 3
AGPtype = "commutator"
norm_type = "ground_state"

grid_size = 1000
append_str = "no_ctrls"

ham = build_ham(
    model_name,
    Ns,
    H_params,
    boundary_conds,
    agp_order,
    norm_type,
    sched,
    symmetries,
    target_symmetries,
)

fname = make_coeffs_fname(
    ham, model_name, ctrls, AGPtype, norm_type, grid_size, sched, append_str
)

# now call function to compute alphas
save_alphas(
    ham, fname, grid_size, sched, agp_order, norm_type, gs_func=ham.get_inst_gstate
)

import matplotlib.pyplot as plt

# do a sanity check
t_data = np.loadtxt("coeffs_data/{0}_alphas_tgrid.txt".format(fname))
alphas_data = np.loadtxt("coeffs_data/{0}_alphas_grid.txt".format(fname))

fig, ax = plt.subplots()
if agp_order == 1:
    ax.plot(t_data, np.abs(alphas_data[:]), label="|a1|")
elif agp_order == 2:
    ax.plot(t_data, np.abs(alphas_data[:, 0]), label="|a1|")
    ax.plot(t_data, np.abs(alphas_data[:, 1]), label="|a2|")
ax.set_yscale("log")
plt.savefig(f"alphas_{agp_order}agp_test.png")
