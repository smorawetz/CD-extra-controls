import os
import sys

import pickle

sys.path.append(os.environ["CD_CODE_DIR"])

from tools.schedules import LinearSchedule, SmoothSchedule


env_dir = os.environ["CD_CODE_DIR"]
with open(f"{env_dir}/dicts/model_param_names.pkl", "rb") as f:
    param_names_dict = pickle.load(f)

norm_type_dict = {"trace": "infT", "ground_state": "zeroT"}
AGPtype_dict = {"commutator": "comm", "krylov": "kry", "chebyshev": "cheby"}

symmetries_names_dict = {"translation_1d": "K{0}", "spin_inversion": "Z{0}"}

scheds_name_dict = {LinearSchedule: "lin", SmoothSchedule: "smooth"}


def make_H_params_str(model_name, H_params):
    return param_names_dict[model_name].format(*H_params)


def make_controls_str(ctrls):
    if ctrls != []:
        return "_".join(ctrls)
    else:
        return "no_controls"


def make_agp_str(AGPtype, norm_type, agp_order):
    if agp_order == 0:
        return "no_agp"
    if AGPtype in AGPtype_dict.keys() and norm_type in norm_type_dict.keys():
        return f"{AGPtype_dict[AGPtype]}_{norm_type_dict[norm_type]}_ord{agp_order}agp"
    elif AGPtype not in AGPtype_dict.keys():
        raise ValueError(f"AGPtype {AGPtype} not recognized")
    elif norm_type not in norm_type_dict.keys():
        raise ValueError(f"norm_type {norm_type} not recognized")


def make_model_str(Ns, model_name, H_params, ctrls):
    H_params_str = make_H_params_str(model_name, H_params)
    ctrls_str = make_controls_str(ctrls)
    return f"{model_name}_N{Ns}_{H_params_str}_{ctrls_str}"


def make_symmetries_str(symmetries):
    if symmetries == []:
        return "no_symm"
    symm_strs = []
    for symmetry in symmetries:  # is a dict of (key: (op, num))
        symm_strs.append(
            symmetries_names_dict[symmetry].format(symmetries[symmetry][1])
        )
    return "_".join(symm_strs)


def gen_fname(model_str, symm_str, agp_str, schedname, grid_size, tau):
    return (
        f"{model_str}_{symm_str}_{agp_str}_{grid_size}steps_{schedname}_sched_tau{tau}"
    )


def make_base_fname(
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
):
    model_str = make_model_str(Ns, model_name, H_params, ctrls)
    symm_str = make_symmetries_str(symmetries)
    agp_str = make_agp_str(AGPtype, norm_type, agp_order)
    sched_name = scheds_name_dict[type(sched)]
    std_name = gen_fname(model_str, symm_str, agp_str, sched_name, grid_size, sched.tau)
    return f"{std_name}_{append_str}"


def make_coeffs_fname(
    ham,
    model_name,
    ctrls,
    AGPtype,
    norm_type,
    grid_size,
    sched,
    append_str,
):
    model_str = make_model_str(ham.Ns, model_name, ham.H_params, ctrls)
    symm_str = make_symmetries_str(ham.symmetries)
    agp_str = make_agp_str(AGPtype, norm_type, ham.agp_order)
    sched_name = scheds_name_dict[type(sched)]
    std_name = gen_fname(model_str, symm_str, agp_str, sched_name, grid_size, sched.tau)
    return f"{std_name}_{append_str}_coeffs"


def make_evolved_wfs_fname(
    ham,
    model_name,
    ctrls,
    AGPtype,
    norm_type,
    grid_size,
    tau,
    append_str,
):
    model_str = make_model_str(ham.Ns, model_name, ham.H_params, ctrls)
    symm_str = make_symmetries_str(ham.symmetries)
    agp_str = make_agp_str(AGPtype, norm_type, ham.agp_order)
    sched = scheds_name_dict[type(ham.schedule)]
    std_name = gen_fname(model_str, symm_str, agp_str, sched, grid_size, tau)
    return f"{std_name}_{append_str}_evolved_wf"


def make_fit_coeffs_fname(AGPtype, agp_order, window_start, window_end):
    return f"universal_fit_{AGPtype}_ord{agp_order}_start{window_start}_end{window_end}"
