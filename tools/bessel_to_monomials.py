import os
import sys

sys.path.append(os.environ["CD_CODE_DIR"])

import numpy as np
import scipy.special as sp


def cmk(m, k, omega0):
    return (
        (-1) ** m
        / sp.factorial(m)
        / sp.factorial(m + 2 * k - 1)
        / (2 * omega0) ** (2 * m + 2 * k - 1)
    )


def buildmat(agp_order, omega0):
    mat = np.zeros((agp_order, agp_order))
    for i in range(agp_order):
        for j in range(i + 1):
            mat[i, j] = cmk(i - j, j + 1, omega0)
    return np.linalg.inv(mat)
