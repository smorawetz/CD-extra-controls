def sin_coupling(t, sched, ns, coeffs):
    """Given some variational coefficients and the mode number they are
    associated with, return the sinusoidal coupling at a given time"""
    lam = sched.get_lam(t)
    return np.sum(coeffs * np.sin(ns * np.pi * lam))


def dlam_sin_coupling(t, sched, ns, coeffs):
    """Given some variational coefficients and the mode number they are
    associated with, return the lambda-derivative of the sinusoidal
    coupling at a given time"""
    lam = sched.get_lam(t)
    return np.sum(np.pi * ns * coeffs * np.cos(np.pi * ns * lam))


def turn_off_coupling(t, sched):
    """Return $1-\lambda(t)$ which is the coupling for H0"""
    return 1 - sched.get_lam(t)


def turn_on_coupling(t, sched):
    """Return $\lambda(t)$ which is the coupling for H1"""
    return sched.get_lam(t)
