import numpy as np


def translation_1d(Ns):
    sites = np.arange(Ns)
    return np.roll(sites, 1)  # shift 1 right


def inversion(Ns):
    sites = np.arange(Ns)
    return -(sites + 1)  # flip


symmetries_dict = {"translation_1d": translation_1d, "spin_inversion": inversion}


def get_symm_op(symm_name, *args):
    return symmetries_dict[symm_name](*args)
