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

symmetries_names_dict = {
    "translation_1d": "K{0}",
    "spin_inversion": "Z{0}",
    "m": "m{0}",
}

scheds_name_dict = {LinearSchedule: "lin", SmoothSchedule: "smooth"}


def make_H_params_str(model_name, H_params):
    return param_names_dict[model_name].format(*H_params)


def make_controls_type_str(ctrls):
    if ctrls != []:
        return "_".join(ctrls)
    else:
        return "no_controls"


def make_model_str(Ns, model_name, H_params, ctrls):
    H_params_str = make_H_params_str(model_name, H_params)
    ctrls_str = make_controls_type_str(ctrls)
    if len(Ns) == 1:
        N_str = f"N{Ns[0]}"
    elif len(Ns) == 2:
        N_str = f"Nx{Ns[0]}_Ny{Ns[1]}"
    else:
        raise ValueError(f"Not built to recognize models with dim > 2")
    return f"{model_name}_{N_str}_{H_params_str}_{ctrls_str}"


def make_symmetries_str(symmetries):
    if symmetries == []:
        return "no_symm"
    symm_strs = []
    for symmetry in symmetries:  # is a dict of (key: (op, num))
        if type(symmetries[symmetry]) == float:  # accounts for mag keyword
            symm_strs.append(
                symmetries_names_dict[symmetry].format(symmetries[symmetry])
            )
        else:
            symm_strs.append(
                symmetries_names_dict[symmetry].format(symmetries[symmetry][1])
            )
    return "_".join(symm_strs)


def make_agp_str(AGPtype, norm_type, agp_order):
    if agp_order == 0:
        return "no_agp"
    if AGPtype in AGPtype_dict.keys() and norm_type in norm_type_dict.keys():
        return f"{AGPtype_dict[AGPtype]}_{norm_type_dict[norm_type]}_ord{agp_order}agp"
    elif AGPtype not in AGPtype_dict.keys():
        raise ValueError(f"AGPtype {AGPtype} not recognized")
    elif norm_type not in norm_type_dict.keys():
        raise ValueError(f"norm_type {norm_type} not recognized")


def make_univ_agp_str(AGPtype, agp_order):
    if agp_order == 0:
        return "no_agp"
    if AGPtype in AGPtype_dict.keys():
        return f"{AGPtype_dict[AGPtype]}_ord{agp_order}agp"
    elif AGPtype not in AGPtype_dict.keys():
        raise ValueError(f"AGPtype {AGPtype} not recognized")


# might need to refactor if have controls other than harmonics, i.e. not sin couplings
def make_ctrls_info_str(ctrls_couplings, ctrls_harmonics, ctrls_coeffs=None):
    if ctrls_coeffs is not None:
        return f"{ctrls_couplings}_harmonics_{ctrls_harmonics}_coeffs_{ctrls_coeffs}"
    else:
        return f"{ctrls_couplings}_harmonics_{ctrls_harmonics}"


def make_file_name(Ns, model_name, H_params, symmetries, ctrls):
    model_str = make_model_str(Ns, model_name, H_params, ctrls)
    symm_str = make_symmetries_str(symmetries)
    return f"{model_str}_{symm_str}"


def make_protocol_name(AGPtype, norm_type, agp_order, grid_size, sched):
    agp_str = make_agp_str(AGPtype, norm_type, agp_order)
    schedname = scheds_name_dict[type(sched)]
    return f"{agp_str}_{grid_size}steps_{schedname}_sched_tau{sched.tau:.6f}"


def make_universal_protocol_name(
    AGPtype, norm_type, agp_order, window_start, window_end, grid_size, sched
):
    agp_str = make_agp_str(
        AGPtype, norm_type, agp_order
    ) + "_window{0:.8f}-{1:.8f}".format(window_start, window_end)
    schedname = scheds_name_dict[type(sched)]
    return f"{agp_str}_{grid_size}steps_{schedname}_sched_tau{sched.tau:.6f}"


def make_fitting_protocol_name(AGPtype, agp_order, sched):
    agp_str = make_univ_agp_str(AGPtype, agp_order)
    schedname = scheds_name_dict[type(sched)]
    return f"{agp_str}_{schedname}_sched_tau{sched.tau:.6f}"


def make_controls_name(ctrls_couplings, ctrls_args):
    if ctrls_couplings == []:
        return "no_controls"
    else:
        Nharmonics = (len(ctrls_args[0]) - 1) // 2
        ctrls_harmonics = [ctrl_args[1 : Nharmonics + 1] for ctrl_args in ctrls_args]
        ctrls_coeffs = [ctrl_args[Nharmonics + 1 :] for ctrl_args in ctrls_args]
        return make_ctrls_info_str(ctrls_couplings, ctrls_harmonics, ctrls_coeffs)


def make_controls_name_no_coeffs(ctrls_couplings, ctrls_args):
    if ctrls_couplings == []:
        return "no_controls"
    else:
        Nharmonics = (len(ctrls_args[0]) - 1) // 2
        ctrls_harmonics = [ctrl_args[1 : Nharmonics + 1] for ctrl_args in ctrls_args]
        return make_ctrls_info_str(ctrls_couplings, ctrls_harmonics, ctrls_coeffs=None)


def make_data_dump_name(
    Ns,
    model_name,
    H_params,
    symmetries,
    sched,
    ctrls,
    ctrls_couplings,
    ctrls_args,
    agp_order,
    AGPtype,
    norm_type,
    grid_size,
):
    file_name = make_file_name(Ns, model_name, H_params, symmetries, ctrls)
    protocol_name = make_protocol_name(AGPtype, norm_type, agp_order, grid_size, sched)
    controls_name = make_controls_name(ctrls_couplings, ctrls_args)
    return file_name, protocol_name, controls_name


def combine_names(*args):
    return "_".join(args)


def make_fit_coeffs_fname(AGPtype, agp_order, window_start, window_end):
    return f"universal_fit_{AGPtype}_ord{agp_order}_start{window_start:.8f}_end{window_end:.8f}"
