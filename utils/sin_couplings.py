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