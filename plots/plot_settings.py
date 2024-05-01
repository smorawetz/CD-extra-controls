import matplotlib.pyplot as plt
import scienceplots


def std_settings():
    params = {
        "text.usetex": True,
        "font.family": "serif",
        "legend.fontsize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 1,
        "patch.edgecolor": "black",
        "pgf.rcfonts": False,
    }
    plt.rcParams.update(params)
    plt.style.use("science")

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    return prop_cycle.by_key()["color"]
