import numpy as np
from scipy.special import kn, kv
from scipy.integrate import quad, trapezoid
from scipy.differentiate import derivative
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const


plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

m_e = (const.m_e.cgs).value
c = (const.c.cgs).value
e = (const.e.esu).value


def calc_N_gam_0(gam, xi, xi_min):
    """
    Calculates initial source term for the number density N_gamma
    """
    theta = (196 / 3) * (xi / xi_min)
    beta = np.sqrt(1 - (1 / (gam**2)))
    return ((gam**2) * beta * np.exp(-gam / theta)) / (theta * kn(2, (1 / theta)))


def calc_E_dot(t, t_0, alpha, B_16):
    """
    Calculates rate of energy injection
    """
    E_b_star = 3e49 * (B_16) ** 2  # erg
    E_dot = (alpha - 1) * (E_b_star / t_0) * ((t / t_0) ** (-alpha))
    return E_dot


def calc_N_e_dot(t, t_0, alpha, B_16, xi, sigma=0.1):
    """
    Rate of change of electron density
    """
    E_dot = calc_E_dot(t, t_0, alpha, B_16)
    return E_dot / (1 + sigma) / xi


def calc_Rn(t, v_n):
    """
    Radius of nebula
    """
    return v_n * t  # returns cm


def calc_EB_and_Bn(t, t_0, alpha, B_16, v_n, sigma):
    """
    Should'nt E_B be E_B_star initially?
    """
    # time array should start exactly when the magnetar is born
    E_B = np.ones_like(t) * 1e50
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


def calc_nu_c(gam, B_n):
    return ((gam**2) * e * B_n) / (2 * np.pi * m_e * c)


def calc_P_nu(nu, nu_c, B_n):
    return ((2 * (e**3) * B_n) / (np.sqrt(3) * m_e * (c**2))) * calc_F(nu / nu_c)


def calc_j_nu(gam, N_gam, P_nu):
    integrand = N_gam * P_nu / 4 / np.pi
    return trapezoid(integrand, gam)


def calc_alpha_nu(gam, nu, N_gam, P_nu):
    d_by_dgam = derivative(N_gam / gam**2, gam)
    integrand = ((gam**2) * P_nu * d_by_dgam) / (8 * np.pi * m_e * (nu**2))
    return trapezoid(integrand, gam)


def calc_L_nu_N_gam_t(t, gam, N_gamma_0, B_16, v_n, sigma):
    N_gam_arr = np.zeros((len(gam), len(t)))
    N_gam_arr[:, 0] = N_gamma_0
    L_nu = np.zeros(len(t))

    beta_sq = 1 - (1 / (gam) ** 2)

    gamma_dot_adiab = np.zeros((len(gam), len(t - 1)))
    gamma_dot_syn = np.zeros((len(gam), len(t - 1)))
    gamma_dot_IC = np.zeros((len(gam), len(t - 1)))
    gamma_dot_brem = np.zeros((len(gam), len(t - 1)))
    E_B, B_n = calc_EB_and_Bn(t, t_0, alpha, B_16, v_n, sigma)

    for i in range(len(t - 1)):
        t_value = t[i + 1]
        for j in range(len(gam)):
            # gamma dot adiabatic
            gamma_dot_adiab[j, i] = -gam[j] * beta_sq[j] * (1 / t[i])

    return N_gam_arr


# Test params

gam = np.logspace(0, 5, 1000)

t = np.logspace(0, 5, 3000)


xi = 5 * u.GeV.to(u.erg)
xi_min = 0.2 * u.GeV.to(u.erg)
alpha = 1.3
B_16 = 1.0
v_n = 3e8
t_0 = 1
sigma = 0.1

nu = 3e20

print(calc_L_nu_N_gam_t(t, gam, calc_N_gam_0(gam, xi, xi_min), B_16, v_n, sigma)[:, 0])


N_gamma_0 = calc_N_gam_0(
    gam,
    xi,
    xi_min,
)
"""
set_xlabel, B_N = calc_EB_and_Bn(t, t_0, alpha, B_16, v_n, sigma)

print(B_N[0])

nu_c = calc_nu_c(gam, B_N[0])

print(nu_c)


P_nu = calc_P_nu(nu, nu_c, B_N[0])

j_nu = calc_j_nu(gam, N_gamma_0, P_nu)

print(j_nu)
"""


# Plotting
"""
fig = plt.figure()

ax = fig.add_subplot(111)


plt.plot(
    t,
    B_n,
)

ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\dot{N_{e}}$")
ax.set_xscale("log")
ax.set_yscale("log")

plt.show()
"""
