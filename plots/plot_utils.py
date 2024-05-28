import matplotlib.pyplot as plt
import numpy as np
import scienceplots


def std_settings():
    params = {
        "text.usetex": True,
        "font.family": "serif",
        "legend.fontsize": 20,
        "axes.labelsize": 28,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 1,
        "patch.edgecolor": "black",
        "pgf.rcfonts": False,
        "font.size": 18,
    }
    plt.rcParams.update(params)
    plt.style.use("science")

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    return prop_cycle.by_key()["color"]


def gauss_broad(xvals, x0, gamma):
    "Gaussian broadening"
    return np.exp(-0.5 * ((xvals - x0) / gamma) ** 2) / np.sqrt(2 * np.pi * gamma**2)
