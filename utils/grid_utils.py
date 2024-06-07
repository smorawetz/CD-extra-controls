from scipy.interpolate import interp1d


def get_coeffs_interp(file_sched, new_sched, file_tgrid, file_coeff_grid):
    """Obtain an interpolation of the coefficients (either Lanczos
    or AGP) from those stored in a file. The file may have a different
    schedule, so this will translate to the new schedule
    Parameters:
        file_sched (Schedule):      Schedule object that the coefficients
                                    were computed for
        new_sched (Schedule):       Schedule object that the coefficients
                                    will be used for
        file_tgrid (np.array):      Time grid of the coefficients, has dims
                                    (grid_size, )
        file_coeff_grid (np.array): Coefficients to interpolate, has dims
                                    (grid_size, agp_order)
    """
    lam_grid = file_sched.get_lam(file_tgrid)
    new_tgrid = new_sched.get_t(lam_grid)
    return interp1d(new_tgrid, file_coeff_grid.T, fill_value="extrapolate")


def get_universal_alphas_func(coeffs_list):
    """Return a function which always returns the same set of coefficients,
    which are presumed to be those obtained for somme "universal" CD driving
    Parameters:
        alphas (np.array):  Coefficients to interpolate, has dims
                            (grid_size, agp_order)
    """
    return lambda x: coeffs_list  # x will be time input, which doesn't matter
