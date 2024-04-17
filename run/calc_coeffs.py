import os
import sys

import numpy as np

sys.path.append(os.environ["CD_CODE_DIR"])

from cd_hamiltonian import Hamiltonian_CD
from tools.build_ham import build_ham
from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op
from utils.file_naming import make_coeffs_fname

EDGE_OFFSET_FACTOR = 1000


## want to save coeffs with standardized grid name, can use to
## then load later and do evolution with easily. probably
## use linear schedule, and do once with a larger grid size
## which should hopefully avoid any problems
def save_alphas(ham, fname, grid_size, sched, agp_order, norm_type, gstate=None):
    """Compute the coefficients for the AGP in the commutator ansatz
    on a grid covering the whole protocol, and save them to a file.
    This can later be indexed and translated for repeated use in
    evolution type problems
    Parameters:
        ham (Hamiltonian_CD):       Counterdiabatic Hamiltonian of interest
        fname (str):                Name of file to store in
        grid_size (int):            Number of time steps to take
        sched (Schedule):           Schedule object that encodes $\lambda(t)$
        agp_order (int):            Order of the AGP to compute
        norm_type (str):            Either "trace" or "ground_state" for the norm
        gstate (np.array):          Ground state of the Hamiltonian, if needed
    """
    lam_grid = np.linspace(0, 1, grid_size)
    t_grid = sched.get_t(lam_grid)
    alphas_grid = np.zeros((grid_size, agp_order))
    for i in range(grid_size):
        t = t_grid[i]
        if i == 0:
            t += (t_grid[1] - t_grid[0]) / EDGE_OFFSET_FACTOR
        elif i == grid_size - 1:
            t -= (t_grid[grid_size - 1] - t_grid[grid_size - 2]) / EDGE_OFFSET_FACTOR
        alphas_grid[i, :] = ham.calc_alphas(t, norm_type, gstate=gstate)
    np.savetxt("coeffs_data/{0}_tgrid.txt".format(fname), t_grid)
    np.savetxt("coeffs_data/{0}_alphas_grid.txt".format(fname), alphas_grid)
    return None


# things to run here
model_name = "TFIM_1D"
Ns = 8
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

agp_order = 2
AGPtype = "commutator"
norm_type = "trace"

grid_size = 1000
append_str = "no_ctrls"

ham = build_ham(
    model_name,
    Ns,
    H_params,
    boundary_conds,
    agp_order,
    sched,
    symmetries,
    target_symmetries,
    norm_type,
)

fname = make_coeffs_fname(
    ham, model_name, ctrls, AGPtype, norm_type, grid_size, tau, append_str
)

# now call function to compute alphas
save_alphas(ham, fname, grid_size, sched, agp_order, norm_type, gstate=None)
