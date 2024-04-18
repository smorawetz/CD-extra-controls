import numpy as np


class LinearSchedule:
    """Linear schedule for $\lambda$(t)"""

    def __init__(self, tau):
        self.tau = tau

    def get_lam(self, t):
        return t / self.tau

    def get_lamdot(self, t):
        return 1 / self.tau

    def get_t(self, lam):
        return lam * self.tau


class SmoothSchedule:
    """Smooth schedule based on previous work"""

    def __init__(self, tau):
        self.tau = tau

    def get_lam(self, t):
        return np.power(
            np.sin(np.pi / 2 * np.power(np.sin(np.pi * t / 2 / self.tau), 2)), 2
        )

    def get_lamdot(self, t):
        lamdot1 = np.sin(np.pi * t / self.tau)
        lamdot2 = np.sin(np.pi * np.power(np.sin(np.pi * t / 2 / self.tau), 2))
        return np.pi * np.pi * lamdot1 * lamdot2 / 4 / self.tau

    def get_t(self, lam):
        return (
            2
            / np.pi
            * np.arcsin(np.sqrt(2 / np.pi * np.arcsin(np.sqrt(lam))))
            * self.tau
        )
