import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op

from scripts.optimize_harmonics_coeffs import optim_harmonic_coeffs

# define the various parameters of the model/task
Ns = 4
model_name = "TFIM_1D"
H_params = [1, 1]  # seed 1 and disorder strength 0.1
boundary_conds = "periodic"

symms = ["translation_1d", "spin_inversion"]
symms_args = [[Ns], [Ns]]
symm_nums = [0, 0]
symmetries = {
    symms[i]: (get_symm_op(symms[i], *symms_args[i]), symm_nums[i])
    for i in range(len(symms))
}
target_symmetries = symmetries

model_kwargs = {}

# schedule will be for coeffs grid, or evolution depending on script
tau = 0.01
sched = SmoothSchedule(tau)

ctrls = ["Hc1", "Hc2"]
ctrls_couplings = ["sin", "sin"]
ctrls_harmonics = [1]

agp_order = 1
AGPtype = "krylov"
norm_type = "trace"

grid_size = 1000

args = (
    ## H params
    Ns,
    model_name,
    H_params,
    boundary_conds,
    symmetries,
    target_symmetries,
    model_kwargs,
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
)

append_str = "std"
maxfields = 3

kwargs = {"append_str": append_str, "maxfields": maxfields}

optim_harmonic_coeffs(*args, **kwargs)
