import os
import sys

import pickle

sys.path.append(os.environ["CD_CODE_DIR"])

from utils.fit_funcs import std_fit_func, cheby_fit_func

fit_funcs_dict = {
    "commutator": std_fit_func,
    "chebyshev": cheby_fit_func,
}

with open("{0}/dicts/fit_funcs.pkl".format(os.environ["CD_CODE_DIR"]), "wb") as f:
    pickle.dump(fit_funcs_dict, f)
