import numpy as np


def translation_1d(Ns):
    sites = np.arange(*Ns)
    return np.roll(sites, 1)  # shift 1 right


def inversion(Ns):
    sites = np.arange(*Ns)
    return -(sites + 1)  # flip


def translation_x_2d(Nx, Ny):
    sites = np.arange(Ns)
    x = sites % Nx
    y = sites // Nx
    return (x + 1) % Nx + Nx * y


def translation_y_2d(Nx, Ny):
    sites = np.arange(Ns)
    x = sites % Nx
    y = sites // Nx
    return x + ((y + 1) % Ny) * Nx


symmetries_dict = {
    "translation_1d": translation_1d,
    "spin_inversion": inversion,
    "translation_x_2d": translation_x_2d,
    "translation_y_2d": translation_y_2d,
}


def get_symm_op(symm_name, *args):
    return symmetries_dict[symm_name](*args)
