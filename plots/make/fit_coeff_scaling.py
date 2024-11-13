import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy

sys.path.append(os.environ["CD_CODE_DIR"])

from plots.plot_utils import std_settings
from tools.calc_universal_fit_coeffs import fit_universal_coeffs

std_settings()

AGPtype = "commutator"
window_start = 0.1
window_end = 10
agp_ords = np.arange(1, 25)
fin_coeffs = []
for agp_order in agp_ords:
    coeffs = fit_universal_coeffs(agp_order, AGPtype, window_start, window_end)
    final_coeff = np.abs(coeffs[-1])
    fin_coeffs.append(final_coeff)

fit_start, fit_end = 1, 13
fit_pts = np.arange(fit_start, fit_end)
asymp = scipy.stats.linregress(fit_pts, np.log(np.array(fin_coeffs))[fit_start:fit_end])

print(asymp.slope)
print(asymp.intercept)

fig, ax = plt.subplots(figsize=(8, 4))
# ax.plot(np.log(np.array(agp_ords)), np.log10(np.array(fin_coeffs)), "o-")
ax.plot(np.array(agp_ords), np.log(np.array(fin_coeffs)), "o-")
ax.plot(
    np.arange(fit_start, fit_end + 1),
    asymp.slope * np.arange(fit_start, fit_end + 1) + np.log(asymp.intercept),
    "--",
)
ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$\log\vert\alpha_\ell\vert$")
plt.savefig("plots/images/final_coeff_fit.pdf")
