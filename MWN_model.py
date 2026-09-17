"""
One zone magnetar wind nebula model of Margalit & Metzger 2018, ApJL 868, L4
(arXiv:1808.09969).  Equation numbers in the docstrings refer to that paper.

Everything is in cgs and every time is in SECONDS.
"""

import shutil

import numpy as np
from scipy.special import kn, kv
from scipy.integrate import quad, trapezoid
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const


# usetex only if there is a latex to call, so that the module always imports
plt.rcParams.update(
    {"text.usetex": shutil.which("latex") is not None, "font.family": "serif"}
)

m_e = (const.m_e.cgs).value
c = (const.c.cgs).value
e = (const.e.esu).value


sigma_t = 6.65 * 10 ** (-25)  # *(u.cm)**2
alpha_fs = 1 / 137
yr = 3.15576e7  # s

# eq. (7) is printed as  dN_gam/dt + d(gam_dot N_gam)/dgam - 3 (Rdot_n/R_n) N_gam
# = N_gam_dot.  N_gam is a number DENSITY, so expansion has to dilute it and the
# sign must be +3, not -3.  Checked: with -1.0 the nebula ends up with ~4e9 times
# the electrons it was ever given, and RM comes out 1e10 too large.
EXPANSION_SIGN = +1.0  # set to -1.0 to reproduce the printed sign of eq. (7)
INCLUDE_IC = True  # inverse Compton branch of eq. (9)


def calc_N_gam_0(gam, xi, xi_min):
    """
    Calculates initial source term for the number density N_gamma

    Relativistic Maxwellian of eq. (6): kT = gam_bar m_e c^2 / 3 = xi / 6, i.e.
    theta = kT/(m_e c^2) = gam_bar/3 = (196/3)(xi/xi_min).
    """
    theta = (196 / 3) * (xi / xi_min)
    beta = np.sqrt(1 - (1 / (gam**2)))
    return ((gam**2) * beta * np.exp(-gam / theta)) / (theta * kn(2, (1 / theta)))


def calc_N_gam_inj(gam, dgam, xi, xi_min):
    """
    Injection shape psi(gam) of the source term of eq. (7): calc_N_gam_0
    renormalised so that sum(psi * dgam) = 1 exactly on the grid, i.e. so that
    N_e_dot = V_n * int N_gam_dot dgam holds to machine precision.
    """
    psi = calc_N_gam_0(gam, xi, xi_min)
    return psi / np.sum(psi * dgam)


def calc_gam_grid(gam_min=1.0, gam_max=1e5, n_gam=500):
    """
    Log spaced Lorentz factor grid, its log step du = d(ln gam), and the exact
    finite volume cell widths (cell edges sit at the geometric means
    sqrt(gam_j gam_j+1), so dgam = 2 gam sinh(du/2) in every cell including the
    two ends).  gam starts at exactly 1, where beta = 0 and eqs. (8) and (9)
    therefore vanish on their own.
    """
    gam = np.logspace(np.log10(gam_min), np.log10(gam_max), n_gam)
    du = np.log(gam[1] / gam[0])
    return gam, du, 2.0 * gam * np.sinh(du / 2.0)


def calc_E_dot(t, t_0, alpha, B_16, E_B_star=None):
    """
    Calculates rate of energy injection (eq. 4).  E_b_star is eq. (1); pass
    E_B_star directly to use the models A/B/C values of the paper instead.
    """
    if E_B_star is None:
        E_B_star = 3e49 * (B_16) ** 2  # erg, eq. (1)
    E_dot = (alpha - 1) * (E_B_star / t_0) * ((t / t_0) ** (-alpha))
    return E_dot


def calc_N_e_dot(t, t_0, alpha, B_16, xi, sigma=0.1, E_B_star=None):
    """
    Rate of change of electron density

    eq. (5).  This is a TOTAL rate (particles per second); the source term of
    eq. (7) is the density rate N_e_dot * psi(gam) / V_n.
    """
    E_dot = calc_E_dot(t, t_0, alpha, B_16, E_B_star)
    return E_dot / (1 + sigma) / xi


def calc_Rn(t, v_n):
    """
    Radius of nebula
    """
    return v_n * t  # returns cm


def calc_EB_and_Bn(t, t_0, alpha, B_16, v_n, sigma, E_B_star=None):
    """
    Magnetic energy E_B and nebula field B_n over the whole time array, eq. (11):
    dE_B/dt = -(Rdot_n/R_n) E_B + sigma/(1+sigma) E_dot, with Rdot_n/R_n = 1/t
    because R_n = v_n t.

    E_B starts at 0 at t[0] = t_0 and is built up by the source term -- E_B_star
    is the reservoir the magnetar draws from, not the nebula's initial content.
    The homogeneous branch of eq. (11) decays as 1/t, so the seed is forgotten
    and E_B/(E_dot t) -> (sigma/(1+sigma))/(2-alpha) at t >> t_0.
    """
    # time array should start exactly when the magnetar becomes active, t = t_0
    E_B = np.zeros_like(t)
    # ensuring everything is in cgs values so we input the erg/s unit instead of Gev/yr
    E_dot = calc_E_dot(t, t_0, alpha, B_16, E_B_star)
    source_term = (sigma / (1 + sigma)) * E_dot
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]  # Time step in seconds
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


def _F_quad(x):
    """
    F(x) of eq. (14) by direct quadrature, used only to build the table below.

    The integral is done in s = ln(y), where the integrand K_{5/3}(e^s) e^s ~
    e^(-2s/3) is smooth and bounded.  Integrating K_{5/3} in y directly, as a
    plain quad(kv, x, np.inf) does, silently returns NEGATIVE values once
    x < ~1e-5 because K_{5/3}(y) ~ y^(-5/3) blows up at the lower limit.
    """

    def fy1(s):
        return kv(5.0 / 3.0, np.exp(s)) * np.exp(s)

    lo = np.log(x)
    I_low = quad(fy1, lo, 0.0, limit=200)[0] if x < 1.0 else 0.0
    return x * (I_low + quad(fy1, max(lo, 0.0), 6.0, limit=200)[0])


# The solver needs ~1e6 evaluations of F per model run, so tabulate it once.
_F_X = np.logspace(-12, 2, 1000)
_F_SPL = CubicSpline(np.log(_F_X), np.log(np.array([_F_quad(x) for x in _F_X])))


def calc_F(x):
    """
    Returns values of function F (defined in eq. 14)
    at x

    Log-log spline of _F_quad inside [1e-12, 1e2] (max relative error 4e-8) and
    the exact asymptotes outside it: F -> 2.1495282 x^(1/3) for x << 1 and
    F -> sqrt(pi/2) x^(1/2) e^(-x) for x >> 1.
    """
    x = np.asarray(x, dtype=float)
    F_x = np.exp(_F_SPL(np.log(np.clip(x, _F_X[0], _F_X[-1]))))
    F_x = np.where(x < _F_X[0], 2.1495282 * x ** (1.0 / 3.0), F_x)
    F_x = np.where(
        x > _F_X[-1],
        np.sqrt(np.pi / 2) * np.sqrt(x) * np.exp(-np.minimum(x, 700.0)),
        F_x,
    )
    return F_x


def calc_nu_c(gam, B_n):
    """
    Characteristic synchrotron frequency, eq. (14)
    """
    return ((gam**2) * e * B_n) / (2 * np.pi * m_e * c)


def calc_P_nu(nu, nu_c, B_n):
    """
    Spectral power of a synchrotron emitting electron, eq. (14)
    """
    return ((2 * (e**3) * B_n) / (np.sqrt(3) * m_e * (c**2))) * calc_F(nu / nu_c)


def calc_j_nu(gam, N_gam, P_nu):
    """
    Synchrotron emissivity, eq. (13)
    """
    integrand = N_gam * P_nu / 4 / np.pi
    return trapezoid(integrand, gam)


def calc_alpha_nu(gam, nu, N_gam, P_nu, du=None):
    """
    Synchrotron self absorption coefficient, eq. (13).

    Note the leading minus sign of eq. (13), and that d/dgam(N_gam/gam^2) is
    taken in u = ln(gam), where the grid is uniform so np.gradient is second
    order: d/dgam = (1/gam) d/du.
    """
    if du is None:
        du = np.log(gam[1] / gam[0])
    d_by_dgam = np.gradient(N_gam / gam**2, du) / gam
    integrand = ((gam**2) * P_nu * d_by_dgam) / (8 * np.pi * m_e * (nu**2))
    return -trapezoid(integrand, gam)


def calc_K_matrix(gam):
    """
    Kernel K_ij = F(nu_c[gam_i]/nu_c[gam_j]) = F((gam_i/gam_j)^2) needed to get
    alpha_nu at every nu_c(gam) for eq. (15).  B_n cancels out of the ratio, so
    this depends on the gam grid alone and is built once per run.
    """
    return calc_F((gam[:, None] / gam[None, :]) ** 2)


def calc_alpha_nu_at_nu_c(gam, du, dgam, N_gam, B_n, K):
    """
    eq. (13) evaluated at nu = nu_c(gam) for every gam at once, as one matvec,
    using gam^2 d/dgam(N_gam/gam^2) dgam = gam d/du(N_gam/gam^2) dgam.
    """
    v = gam * np.gradient(N_gam / gam**2, du) * dgam
    nu_c = calc_nu_c(gam, B_n)
    P_pref = (2 * (e**3) * B_n) / (np.sqrt(3) * m_e * (c**2))  # eq. (14)
    return -(P_pref * (K @ v)) / (8 * np.pi * m_e * (nu_c**2))


def calc_tau_gamma(alpha_nu_at_nu_c, R_n):
    """
    Synchrotron optical depth seen by an electron of Lorentz factor gam, eq. (15)
    """
    return alpha_nu_at_nu_c * R_n


def calc_fssa(tau_gamma):
    """
    Self absorption suppression of the synchrotron cooling rate, eq. (15)
    """
    tau_gamma = np.maximum(tau_gamma, 1e-30)  # optically thin limit is f_ssa -> 1
    return (1 - np.exp(-tau_gamma)) / (tau_gamma)


def calc_L_nu(j_nu, alpha_nu, R_n):
    """
    Synchrotron luminosity of the nebula, eq. (12)
    """
    return (
        4
        * ((np.pi) ** 2)
        * (R_n**2)
        * (j_nu / alpha_nu)
        * (1 - np.exp(-alpha_nu * R_n))
    )


def calc_gamma_dot_adiab(gam, t):
    """
    Adiabatic cooling, eq. (8), with Rdot_n/R_n = 1/t
    """
    return -gam * (1 - 1 / gam**2) * (1 / t)


def calc_gamma_dot_syn_IC(gam, U):
    """
    Synchrotron and inverse Compton cooling, eq. (9).  U is the energy density
    the electrons lose to: f_ssa(gam) B_n^2/(8 pi) for synchrotron and
    L_rad/(4 pi c R_n^2) for inverse Compton.
    """
    return -(4 / 3) * (sigma_t / (m_e * c)) * (1 - 1 / gam**2) * (gam**2) * U


def calc_gamma_dot_brem(gam, n_e):
    """
    Bremsstrahlung cooling, eq. (10).

    Eq. (10) is the ultra relativistic limit and stays finite at gam = 1, which
    would cool electrons straight off the bottom of the grid; the extra beta
    restores the v/c scaling so gamma_dot -> 0 as gam -> 1.
    """
    beta = np.sqrt(1 - 1 / gam**2)
    return -(5 / 3) * c * sigma_t * alpha_fs * n_e * gam ** (1.2) * beta


def calc_L_rad(dgam, N_gam, gamma_dot_syn, V_n):
    """
    Synchrotron luminosity of the whole nebula, i.e. its synchrotron energy loss
    rate.  Identical to int L_nu dnu but free, since gamma_dot_syn has just been
    computed; it feeds the inverse Compton branch of eq. (9) on the next step.
    """
    return -m_e * (c**2) * V_n * np.sum(N_gam * gamma_dot_syn * dgam)


def calc_RM(gam, dgam, N_gam, R_n, B_n, lam=None):
    """
    Rotation measure through the nebula, eq. (16), with lam = R_n by default.

    The cgs prefactor e^3/(2 pi m_e^2 c^4) gives rad cm^-2, hence the 1e4 to
    return the usual rad m^-2.
    """
    if lam is None:
        lam = R_n
    K_RM = (e**3) / (2 * np.pi * (m_e**2) * (c**4))
    return 1e4 * K_RM * np.sqrt(lam / R_n) * R_n * B_n * np.sum(N_gam / gam**2 * dgam)


def calc_L_nu_spectrum(gam, du, nu_arr, N_gam, B_n, R_n):
    """
    L_nu on a frequency grid, eqs. (12) to (14)
    """
    nu_c = calc_nu_c(gam, B_n)
    L_nu = np.zeros(len(np.atleast_1d(nu_arr)))
    for i, nu in enumerate(np.atleast_1d(nu_arr)):
        P_nu = calc_P_nu(nu, nu_c, B_n)
        j_nu = calc_j_nu(gam, N_gam, P_nu)
        alpha_nu = calc_alpha_nu(gam, nu, N_gam, P_nu, du)
        L_nu[i] = calc_L_nu(j_nu, alpha_nu, R_n)
    return L_nu


def calc_L_nu_obs(nu_obs, F_nu_obs, z=0.1927, D_L_Mpc=972.0):
    """
    Observed flux density -> rest frame frequency and luminosity for FRB 121102:
    nu_emit = (1+z) nu_obs and L_nu = 4 pi D_L^2 F_nu (1+z).  This is the
    convention that reproduces the paper's quoted L_nu = 2.7e29 erg/s/Hz at
    7.2 GHz for the 6 GHz VLA point of Chatterjee et al. (2017).
    """
    D_L = D_L_Mpc * 1e6 * (const.pc.cgs).value
    return (1 + z) * nu_obs, 4 * np.pi * (D_L**2) * F_nu_obs * (1 + z)


def calc_L_nu_N_gam_t(
    t,
    gam,
    du,
    dgam,
    nu_arr,
    t_0,
    alpha,
    B_16,
    xi,
    xi_min,
    v_n,
    sigma,
    E_B_star=None,
    snap_t=(),
):
    """
    Evolves the electron number density N_gam of eq. (7) over the time array t
    (seconds, starting at t_0) and returns, at every step, the rotation measure
    (eq. 16) and the light curves L_nu(t) at the frequencies in nu_arr (eq. 12).

    Returns (snaps, RM, L_nu, N_tot, B_n, R_n), where snaps is one
    (t, N_gam, B_n, R_n) tuple per requested epoch in snap_t, for spectra.
    """
    E_B, B_n = calc_EB_and_Bn(t, t_0, alpha, B_16, v_n, sigma, E_B_star)  # eq. (11)
    R_n = calc_Rn(t, v_n)
    V_n = (4 / 3) * np.pi * R_n**3

    K = calc_K_matrix(gam)  # eq. (15), grid only, so build it once
    psi = calc_N_gam_inj(gam, dgam, xi, xi_min)  # eq. (6)

    nu_arr = np.atleast_1d(nu_arr)
    snap_t = np.atleast_1d(snap_t) if len(snap_t) else np.array([])
    snaps = [None] * len(snap_t)

    N_gam = np.zeros_like(gam)
    RM = np.zeros_like(t)
    N_tot = np.zeros_like(t)
    L_nu = np.zeros((len(nu_arr), len(t)))
    L_rad = 0.0

    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]

        # self absorption suppression of the cooling rate, eqs. (13) and (15)
        alpha_nu_at_nu_c = calc_alpha_nu_at_nu_c(gam, du, dgam, N_gam, B_n[i], K)
        fssa = calc_fssa(calc_tau_gamma(alpha_nu_at_nu_c, R_n[i]))

        # cooling terms, eqs. (8) to (10)
        gamma_dot_adiab = calc_gamma_dot_adiab(gam, t[i])
        gamma_dot_syn = calc_gamma_dot_syn_IC(gam, fssa * (B_n[i] ** 2) / (8 * np.pi))
        gamma_dot_IC = (
            calc_gamma_dot_syn_IC(gam, L_rad / (4 * np.pi * c * R_n[i] ** 2))
            if INCLUDE_IC
            else 0.0
        )
        gamma_dot_brem = calc_gamma_dot_brem(gam, np.sum(N_gam * dgam))
        gamma_dot = gamma_dot_adiab + gamma_dot_syn + gamma_dot_IC + gamma_dot_brem

        # seed photon field for the next step's inverse Compton term
        L_rad = calc_L_rad(dgam, N_gam, gamma_dot_syn, V_n[i])

        # eq. (7).  Every gamma_dot is negative, so a cell can only be fed by the
        # cell above it: backward Euler in gam is upper bidiagonal and solves in a
        # single pass.  It is unconditionally stable, which is what makes this
        # tractable -- synchrotron cooling near t_0 would otherwise force
        # dt/t < 1e-7.  Where |gamma_dot| dt/dgam >> 1 the update relaxes to
        # N_gam -> (flux in)/|gamma_dot|, exactly the cooled pile up the paper
        # describes.  Expansion is applied as the exact factor (t_prev/t)^3 rather
        # than a discretised 3 dt/t, which conserves particles to machine precision.
        gamma_dot_abs = np.abs(gamma_dot)
        gamma_dot_abs[0] = 0.0  # zero flux floor: nothing cools below gam = 1
        N_gam_dot = calc_N_e_dot(t[i], t_0, alpha, B_16, xi, sigma, E_B_star) * psi
        S = N_gam * (t[i - 1] / t[i]) ** (3.0 * EXPANSION_SIGN) + N_gam_dot / V_n[i] * dt
        ab = np.zeros((2, len(gam)))
        ab[0, 1:] = -dt * gamma_dot_abs[1:] / dgam[:-1]
        ab[1] = 1.0 + dt * gamma_dot_abs / dgam
        N_gam = solve_banded((0, 1), ab, S)

        RM[i] = calc_RM(gam, dgam, N_gam, R_n[i], B_n[i])  # eq. (16)
        N_tot[i] = V_n[i] * np.sum(N_gam * dgam)
        L_nu[:, i] = calc_L_nu_spectrum(gam, du, nu_arr, N_gam, B_n[i], R_n[i])

        for k in np.where((t[i - 1] < snap_t) & (snap_t <= t[i]))[0]:
            snaps[k] = (t[i], N_gam.copy(), B_n[i], R_n[i])

    return snaps, RM, L_nu, N_tot, B_n, R_n


if __name__ == "__main__":
    # Model A of the paper: E_B_star = 5e50 erg, t_0 = 0.2 yr, v_n = 3e8 cm/s,
    # alpha = 1.3, chi = 0.2 GeV, sigma = 0.1, giving t_age ~ 12.4 yr.
    gam, du, dgam = calc_gam_grid(1.0, 1e5, 500)

    xi = 0.2 * u.GeV.to(u.erg)
    xi_min = 0.2 * u.GeV.to(u.erg)
    alpha = 1.3
    B_16 = 1.0
    E_B_star = 5e50
    v_n = 3e8
    t_0 = 0.2 * yr
    sigma = 0.1
    t_age = 12.4 * yr

    t = np.logspace(np.log10(t_0), np.log10(3.5 * t_age), 3000)
    nu = np.array([3e9])

    snaps, RM, L_nu, N_tot, B_n, R_n = calc_L_nu_N_gam_t(
        t,
        gam,
        du,
        dgam,
        nu,
        t_0,
        alpha,
        B_16,
        xi,
        xi_min,
        v_n,
        sigma,
        E_B_star=E_B_star,
        snap_t=[t_age],
    )

    t_snap, N_gam_snap, B_snap, R_snap = snaps[0]
    nu_arr = np.logspace(7, 13, 150)
    L_nu_snap = calc_L_nu_spectrum(gam, du, nu_arr, N_gam_snap, B_snap, R_snap)

    print("model A at t = %.2f yr" % (t_snap / yr))
    print("  B_n     = %.4f G" % B_snap)
    print("  R_n     = %.3e cm" % R_snap)
    print("  N_tot   = %.3e electrons" % (4 / 3 * np.pi * R_snap**3 * np.sum(N_gam_snap * dgam)))
    print("  RM      = %.3e rad m^-2   (observed 1.46e5)" % np.interp(t_age, t, RM))
    print("  L_nu max= %.3e erg/s/Hz at %.2f GHz" % (L_nu_snap.max(), nu_arr[L_nu_snap.argmax()] / 1e9))

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.loglog(t / yr, RM)
    ax1.axhline(1.46e5, ls="--", color="k")
    ax1.set_xlabel(r"$t$ (yr)")
    ax1.set_ylabel(r"RM (rad m$^{-2}$)")

    ax2.loglog(nu_arr / 1e9, L_nu_snap)
    ax2.set_xlabel(r"$\nu$ (GHz)")
    ax2.set_ylabel(r"$L_{\nu}$ (erg s$^{-1}$ Hz$^{-1}$)")

    fig.tight_layout()
    plt.show()
