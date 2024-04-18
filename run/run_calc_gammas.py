import os
import sys

import numpy as np
import scipy

sys.path.append(os.environ["CD_CODE_DIR"])

from tools.calc_coeffs import save_lanc_coeffs
from tools.calc_coeffs import save_gammas
from tools.schedules import LinearSchedule
from tools.build_ham import build_ham
from tools.symmetries import get_symm_op
from utils.file_naming import make_coeffs_fname
from utils.grid_utils import get_coeffs_interp

from agp.krylov_construction import op_norm

# things to run here
model_name = "TFIM_1D"
Ns = 14
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

agp_order = 1
AGPtype = "krylov"
norm_type = "trace"

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
lanc_tgrid, lanc_grid = save_lanc_coeffs(
    ham, fname, grid_size, sched, agp_order, norm_type, gstate=None
)
ham.lanc_interp = scipy.interpolate.interp1d(lanc_tgrid, lanc_grid, axis=0)
tgrid, gammas_grid = save_gammas(
    ham, fname, grid_size, sched, agp_order, norm_type, gstate=None
)
ham.gammas_interp = scipy.interpolate.interp1d(tgrid, gammas_grid, axis=0)

import matplotlib.pyplot as plt

# do a sanity check
t_data = np.loadtxt("coeffs_data/{0}_gammas_tgrid.txt".format(fname))
gammas_data = np.loadtxt("coeffs_data/{0}_gammas_grid.txt".format(fname))

fig, ax = plt.subplots()
if agp_order == 1:
    ax.plot(t_data, gammas_data[:], label="g1")
elif agp_order == 2:
    ax.plot(t_data, gammas_data[:, 0], label="g1")
    ax.plot(t_data, gammas_data[:, 1], label="g2")
plt.savefig(f"gammas_{agp_order}agp_test.png")

print(ham.lanc_interp(0.5 * tau))
