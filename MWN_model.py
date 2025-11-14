import numpy as np
from scipy.special import kn
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


def calc_N_gamma_0(gamma, xi, xi_min=0.2):
    theta = (196 / 3) * (xi / xi_min)
    beta = np.sqrt(1 - (1 / (gamma**2)))
    return ((gamma**2) * beta * np.exp(-gamma / theta)) / (theta * kn(2, (1 / theta)))


def calc_E_dot(t, t_0, alpha, B_16):
    E_b_star = 3e49 * (B_16) ** 2  # erg
    E_dot = (
        (alpha - 1)
        * (E_b_star / t_0)
        * ((t / t_0) ** (-alpha))
        * (u.erg / u.year).to(u.GeV / u.year)
    )
    return E_dot


def calc_N_e_dot(t, t_0, alpha, B_16, xi, sigma=0.1):
    E_dot = calc_E_dot(t, t_0, alpha, B_16)
    return E_dot / (1 + sigma) / xi


gamma = np.logspace(0, 3, 1000)

t = np.logspace(0, 5, 3000)

xi = 5

fig = plt.figure()

ax = fig.add_subplot(111)

plt.plot(t, calc_N_e_dot(t, 0.1, 1.5, 1, xi))
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\dot{N_{e}}$")
ax.set_xscale("log")
ax.set_yscale("log")

plt.show()
