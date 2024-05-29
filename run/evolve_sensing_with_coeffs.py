import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import quspin

from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op
from utils.file_naming import make_base_fname

from scripts.calc_commutator_coeffs import calc_comm_coeffs
from scripts.calc_krylov_coeffs import calc_kry_coeffs
from scripts.iterative_GS_optimization import do_iterative_evolution
from scripts.optimize_harmonics_coeffs import optim_harmonic_coeffs
from scripts.time_evolution import run_time_evolution

###############################################################
## this is a generic example of how to run a pre made script ##
###############################################################

g = 0.01

# define the various parameters of the model/task
Ns = 3
model_name = "Field_Sensing_1D"
H_params = [1, 1, g]
boundary_conds = "periodic"

symms = ["translation_1d"]
symms_args = [[Ns]]
symm_nums = [0]
symmetries = {
    symms[i]: (get_symm_op(symms[i], *symms_args[i]), symm_nums[i])
    for i in range(len(symms))
}
target_symmetries = symmetries

# schedule will be for coeffs grid, or evolution depending on script
evolve_tau = 0.01
coeffs_tau = 1
evolve_sched = SmoothSchedule(evolve_tau)
coeffs_sched = SmoothSchedule(coeffs_tau)

ctrls = []
ctrls_couplings = []
ctrls_args = []

agp_order = 1
AGPtype = "krylov"
norm_type = "trace"

grid_size = 1000

# have generic list of args that get used for every function
args = (
    ## H params
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    target_symmetries,
    ## schedule params
    evolve_tau,
    evolve_sched,
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
)

# now load the coefficients for the local GHZ state prep

coeffs_model_name = "TFIM_1D"
coeffs_H_params = [1, 1]
coeffs_symms = ["translation_1d", "spin_inversion"]
coeffs_symms_args = [[Ns], [Ns]]
coeffs_symm_nums = [0, 0]
coeffs_symmetries = {
    coeffs_symms[i]: (
        get_symm_op(coeffs_symms[i], *coeffs_symms_args[i]),
        coeffs_symm_nums[i],
    )
    for i in range(len(coeffs_symms))
}

coeffs_append_str = "std"

fname = make_base_fname(
    Ns,
    coeffs_model_name,
    coeffs_H_params,
    coeffs_symmetries,
    ctrls,
    agp_order,
    AGPtype,
    norm_type,
    grid_size,
    coeffs_sched,
    coeffs_append_str,
)
coeffs_fname = fname + "_coeffs"

wfs_append_str = "g{0:.6f}".format(g)

kwargs = {
    "save_wf": False,
    "coeffs_fname": coeffs_fname,
    "coeffs_sched": coeffs_sched,
    "wfs_save_append_str": wfs_append_str,
}

final_wf = run_time_evolution(*args, **kwargs)

# calculate Sz op with QuSpin using same symmetries as model
Szterms = [["z", [[1, i] for i in range(Ns)]]]
basis = quspin.basis.spin_basis_general(Ns, S="1/2", **symmetries)
checks = {"check_herm": False, "check_symm": False}
Szop = quspin.operators.hamiltonian(Szterms, [], basis=basis, **checks)
mag = Szop.matrix_ele(final_wf, final_wf).real  # is guaranteed to have 0 imag part

mag_data_fname = "{0}/plots/data/{1}_sensing_mag.txt".format(
    os.environ["CD_CODE_DIR"], fname
)
data_file = open(mag_data_fname, "a")
data_file.write("{0}\t{1}\n".format(g, mag))
data_file.close()
