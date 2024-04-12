from scipy.interpolate import interp1d


def get_coeffs_interp(file_sched, new_sched, fname):
    """Obtain an interpolation of the coefficients (either Lanczos
    or AGP) from those stored in a file. The file may have a different
    schedule, so this will translate to the new schedule
    Parameters:
        file_sched (Schedule):      Schedule object that the coefficients
                                    were computed for
        new_sched (Schedule):       Schedule object that the coefficients
                                    will be used for
        fname (str):                Name of file to read coefficients from
    """
    coeffs_grid = np.loadtxt(fname)  # shape (grid_size, agp_order)
    file_tgrid = np.linspace(0, file_sched.tau, coeffs_grid.shape[0])
    lam_grid = file_sched.get_lam(file_tgrid)
    new_tgrid = new_sched.get_t(lam_grid)
    return interp1d(new_tgrid, coeffs_grid)
