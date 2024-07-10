import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import numpy as np
from scipy.optimize import curve_fit

from tools.calc_universal_fit_coeffs import fit_universal_coeffs, std_fit_func
from utils.file_naming import make_fit_coeffs_fname

agp_order = 10
window_start = 0.2
window_end = 1.0

fit_func = std_fit_func
AGPtype = "commutator"

alphas = fit_universal_coeffs(fit_func, agp_order, window_start, window_end)
fname = make_fit_coeffs_fname(AGPtype, agp_order, window_start, window_end)

np.savetxt("{0}/coeffs_data/{1}.txt".format(os.environ["CD_CODE_DIR"], fname), alphas)
