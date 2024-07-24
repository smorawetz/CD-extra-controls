import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import numpy as np

from tools.schedules import LinearSchedule, SmoothSchedule
from tools.symmetries import get_symm_op

from scripts.calc_commutator_coeffs import calc_comm_coeffs
from scripts.calc_krylov_coeffs import calc_kry_coeffs
from scripts.iterative_GS_optimization import do_iterative_evolution
from scripts.optimize_harmonics_coeffs import optim_harmonic_coeffs
from scripts.time_evolution import run_time_evolution

taskid = int(os.getenv("SGE_TASK_ID"))

listof_Ns = np.arange(2, 14 + 1, 1)
listof_agp_orders = listof_Ns // 2

Ns = listof_Ns[taskid - 1]
agp_order = listof_agp_orders[taskid - 1]

# define the various parameters of the model/task
model_name = "TFIM_1D"
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

model_kwargs = {}

# schedule will be for coeffs grid, or evolution depending on script
tau = 1
sched = SmoothSchedule(tau)

ctrls = []
ctrls_couplings = []
ctrls_args = []

AGPtype = "krylov"
# AGPtype = "commutator"
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
    model_kwargs,
    ## schedule params
    tau,
    sched,
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

# now define the kwargs which are specific to each script
append_str = "std"

kwargs = {"append_str": append_str}

calc_kry_coeffs(*args, **kwargs)
# calc_comm_coeffs(*args, **kwargs)
