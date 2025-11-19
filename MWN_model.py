import numpy as np
from scipy.special import kn, kv
from scipy.integrate import quad
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

m_e = (const.m_e.cgs).value
c = (const.c.cgs).value
e = (const.e.esu).value


def calc_N_gamma_0(gamma, xi, xi_min=0.2):
    theta = (196 / 3) * (xi / xi_min)
    beta = np.sqrt(1 - (1 / (gamma**2)))
    return ((gamma**2) * beta * np.exp(-gamma / theta)) / (theta * kn(2, (1 / theta)))


def calc_E_dot(t, t_0, alpha, B_16):
    E_b_star = 3e49 * (B_16) ** 2  # erg
    E_dot = (alpha - 1) * (E_b_star / t_0) * ((t / t_0) ** (-alpha))
    return E_dot


def calc_N_e_dot(t, t_0, alpha, B_16, xi, sigma=0.1):
    E_dot = calc_E_dot(t, t_0, alpha, B_16)
    return E_dot / (1 + sigma) / xi


def calc_Rn(t, v_n):
    return v_n * t  # returns cm


def calc_EB_and_Bn(t, t_0, alpha, B_16, v_n, sigma):
    # time array should start exactly when the magnetar is born
    E_B = np.zeros_like(t)
    # ensuring everything is in cgs values so we input the erg/year unit instead of Gev/year
    E_dot = calc_E_dot(t, t_0, alpha, B_16)
    source_term = (sigma / (1 + sigma)) * E_dot
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]  # Time step in years
        expansion_loss_rate = 1.0 / t[i - 1]
        # dE = (Source - Loss) * dt
        loss = expansion_loss_rate * E_B[i - 1]
        dE = (source_term[i - 1] - loss) * dt

        E_B[i] = E_B[i - 1] + dE
    R_n = calc_Rn(t, v_n)

    # Calculate Magnetic Field B_n from Energy E_B
    # B_n = sqrt(6 * E_B / R_n^3)
    B_n = np.sqrt(6 * E_B / (R_n**3))

    return E_B, B_n


def calc_F(x):
    """
    Returns values of function F (defined in eq. 14)
    at x
    """

    def fy1(y):
        return kv(5.0 / 3.0, y)

    if isinstance(x, float):
        return x * quad(fy1, x, np.inf)[0]
    else:
        F_x = np.zeros(len(x))
        for i, x_i in enumerate(x):
            F_x[i] = x_i * quad(fy1, x_i, np.inf)[0]
        return F_x


def calc_nu_c(gamma, t, t_0, alpha, B_16, v_n, sigma):
    E_B, B_n = calc_EB_and_Bn(t, t_0, alpha, B_16, v_n, sigma)
    return ((gamma**2) * e * B_n) / (2 * np.pi * m_e * c)


def calc_P_nu(gamma, t, nu, t_0, alpha, B_16, v_n, sigma):
    E_B, B_n = calc_EB_and_Bn(t, t_0, alpha, B_16, v_n, sigma)
    nu_c = calc_nu_c(gamma, t, t_0, alpha, B_16, v_n, sigma)
    return ((2 * (e**3) * B_n) / (np.sqrt(3) * m_e * (c**2))) * calc_F(nu / nu_c)


# Test params

gamma = np.logspace(0, 3, 1000)

t = np.logspace(0, 5, 3000)

xi = 5 * u.GeV.to(u.erg)
alpha = 1.3
B_16 = 1.0
v_n = 3e8
t_0 = 1e-5
sigma = 0.1

# Plotting

fig = plt.figure()

ax = fig.add_subplot(111)

E_B, B_n = calc_EB_and_Bn(t, t_0, alpha, B_16, v_n, sigma)

plt.plot(
    t,
    B_n,
)

ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\dot{N_{e}}$")
ax.set_xscale("log")
ax.set_yscale("log")

plt.show()
