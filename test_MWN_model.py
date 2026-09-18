"""
Tests MWN_model.py against Margalit & Metzger 2018 (arXiv:1808.09969):
the exact limits of F(x), the analytic scalings of eqs. (17) - (19), and the
numerical results of Figs. 3 and 4 (RM(t) -> t_age, spectra, light curves).

    python test_MWN_model.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.integrate import trapezoid

import MWN_model as M
from MWN_model import yr

RM_OBS = 1.46e5  # rad m^-2, Michilli et al. 2018, MJD 57747

# Models A, B and C of the paper (chi = 0.2 GeV, sigma = 0.1, lam = R_n throughout)
MODELS = {
    "A": dict(E_B_star=5e50, t_0=0.2 * yr, v_n=3e8, alpha=1.30, t_age=12.4 * yr),
    "B": dict(E_B_star=5e50, t_0=0.6 * yr, v_n=1e8, alpha=1.30, t_age=37.8 * yr),
    "C": dict(E_B_star=4.9e51, t_0=0.2 * yr, v_n=9e8, alpha=1.83, t_age=13.1 * yr),
}
XI = 0.2 * u.GeV.to(u.erg)
SIGMA = 0.1

# Persistent source of FRB 121102 (Chatterjee et al. 2017), observed frame
DATA_NU_OBS = np.array([1.63, 3.0, 6.0, 10.0, 15.0, 22.0]) * 1e9
DATA_F_OBS = np.array([250, 206, 203, 166, 103, 66]) * 1e-29  # erg/s/cm^2/Hz

_results = []


def check(name, got, want, tol, note=""):
    """
    Records one relative-tolerance check (absolute if want == 0).
    """
    ok = abs(got - want) <= tol * (abs(want) if want != 0 else 1.0)
    _results.append((ok, name, got, want, note))
    return ok


def check_fac(name, got, want, fac, note=""):
    """
    Records one order-of-magnitude check: passes if 1/fac <= got/want <= fac.
    Used where the paper only gives a value read off a figure.
    """
    ok = 1.0 / fac <= got / want <= fac
    _results.append((ok, name, got, want, note + " (within x%g)" % fac))
    return ok


def run(key, n_gam=500, n_t=3000, nu_arr=(3e9,), t_max_fac=3.5):
    """
    Runs one of the paper's models and returns everything needed by the tests.
    """
    p = MODELS[key]
    gam, du, dgam = M.calc_gam_grid(1e-6, 1e5, n_gam)
    t = np.logspace(np.log10(p["t_0"]), np.log10(t_max_fac * p["t_age"]), n_t)
    snap_t = np.array([p["t_age"] / 3, p["t_age"], 3 * p["t_age"]])
    out = M.calc_L_nu_N_gam_t(
        t, gam, du, dgam, np.atleast_1d(nu_arr), p["t_0"], p["alpha"], 1.0,
        XI, XI, p["v_n"], SIGMA, E_B_star=p["E_B_star"], snap_t=snap_t,
    )
    snaps, RM, L_nu, N_tot, B_n, R_n = out
    return dict(p, key=key, gam=gam, du=du, dgam=dgam, t=t, snaps=snaps,
                RM=RM, L_nu=L_nu, N_tot=N_tot, B_n=B_n, R_n=R_n)


def crossing(t, RM, level=RM_OBS):
    """
    Epoch at which RM falls back through level, by log-log interpolation.  RM
    rises to a peak and then declines, so only the falling branch is used.
    """
    i = np.argmax(RM)
    return np.exp(np.interp(np.log(level), np.log(RM[:i:-1]), np.log(t[:i:-1])))


# ------------------------------------------------------------------
# 1-3.  F(x): eq. (14) against its exact limits
# ------------------------------------------------------------------
xs = np.logspace(-12, 2, 60)
check("F(x) vs quadrature, 1e-12..1e2",
      np.max(np.abs(M.calc_F(xs) / np.array([M._F_quad(x) for x in xs]) - 1)), 0.0, 1e-6)
check("F small-x -> 2.1495282 x^(1/3)",
      float(M.calc_F(1e-14)) / (2.1495282 * 1e-14 ** (1 / 3)), 1.0, 1e-4)
xf = np.logspace(-4, 0.5, 20001)
check("F peak value (0.9180123)", float(M.calc_F(xf).max()), 0.9180123, 1e-4)
check("F peak position (0.2858122)", float(xf[M.calc_F(xf).argmax()]), 0.2858122, 1e-3)
xi_ = np.logspace(-12, np.log10(60), 400001)
check("int F dx = 8 pi / 9 sqrt(3)",
      float(trapezoid(M.calc_F(xi_), xi_)), 8 * np.pi / (9 * np.sqrt(3)), 1e-4)

# ------------------------------------------------------------------
# 4.  alpha_nu at nu_c: the matvec of eq. (13) vs the plain per-gamma loop
# ------------------------------------------------------------------
g, dulog, dg = M.calc_gam_grid(1e-6, 1e4, 60)
N_test = g**2 * np.exp(-g / 65.0)
B_test = 0.14
a_mat = M.calc_alpha_nu_at_nu_c(g, dulog, dg, N_test, B_test, M.calc_K_matrix(g))
nu_c_test = M.calc_nu_c(g, B_test)
a_loop = np.array([
    M.calc_alpha_nu(g, nu, N_test, M.calc_P_nu(nu, nu_c_test, B_test), dulog)
    for nu in nu_c_test
])
check("alpha_nu matvec vs per-gamma loop",
      float(np.max(np.abs(a_mat / a_loop - 1))), 0.0, 3e-2,
      "matvec sums with dgam, the loop trapezoids in gam; the two quadrature "
      "rules differ by ~2% on the strongly non-uniform kinetic-energy grid")

# ------------------------------------------------------------------
# 5.  eq. (16) prefactor, including the cgs -> rad m^-2 conversion.  Put every
#     electron in the bottom cell, so RM must reduce to
#     1e4 * e^3/(2 pi m_e^2 c^4) * R_n * B_n * n_e / gam_bottom^2.
# ------------------------------------------------------------------
g2, du2, dg2 = M.calc_gam_grid(1e-6, 1e5, 200)
N_delta = np.zeros_like(g2)
N_delta[0] = 1.0 / dg2[0]  # n_e = 1 cm^-3, all of it in the bottom cell
check("calc_RM prefactor, eq.(16)",
      float(M.calc_RM(g2, dg2, N_delta, 1e17, 0.24)),
      1e4 * M.e**3 / (2 * np.pi * M.m_e**2 * M.c**4) * 1e17 * 0.24 / g2[0] ** 2, 1e-12)

# ------------------------------------------------------------------
# Model runs
# ------------------------------------------------------------------
print("running models A, B, C ...")
runs = {k: run(k, nu_arr=(325e6, 1.4e9, 3e9)) for k in ("A", "B", "C")}

for k, r in runs.items():
    a, t_age = r["alpha"], r["t_age"]
    i_age = np.argmin(np.abs(r["t"] - t_age))
    R17 = r["R_n"][i_age] / 1e17
    Edot_t50 = M.calc_E_dot(t_age, r["t_0"], a, 1.0, r["E_B_star"]) * t_age / 1e50

    # 5.  particle conservation: the solver of eq. (7) must neither create nor
    #     destroy electrons, so N_tot has to equal everything eq. (5) injected
    N_inj = np.sum(
        M.calc_N_e_dot(r["t"][1:i_age + 1], r["t_0"], a, 1.0, XI, SIGMA, r["E_B_star"])
        * np.diff(r["t"][:i_age + 1])
    )
    check("%s  particle conservation, eq.(7)" % k, r["N_tot"][i_age] / N_inj, 1.0, 1e-6)

    # 6.  exact solution of eq. (11):
    #     E_B = (sigma/(1+sigma)) E_dot t / (2-alpha) * [1 - (t_0/t)^(2-alpha)]
    E_B_num = r["B_n"][i_age] ** 2 * r["R_n"][i_age] ** 3 / 6
    check("%s  E_B vs exact eq.(11)" % k, E_B_num / (Edot_t50 * 1e50),
          (SIGMA / (1 + SIGMA)) / (2 - a) * (1 - (r["t_0"] / t_age) ** (2 - a)), 0.01)

    # 7.  eq. (17):  B_n ~ 0.24 G sigma_-1^1/2 R17^-3/2 (Edot t)_50^1/2.  This is
    #     a scaling only: it assumes E_B = (sigma/(1+sigma)) E_dot t, i.e. it drops
    #     the 1/(2-alpha) and the transient of eq. (11), which for alpha = 1.83
    #     (model C) is a factor 1.7 in B_n.
    check_fac("%s  B_n vs eq.(17)" % k, r["B_n"][i_age],
              0.24 * R17**-1.5 * Edot_t50**0.5, 2.0)

    # 8.  eq. (18):  N_0 ~ 4e53 chi_0.2^-1 E_50  (paper uses the FULL E_B_star)
    check_fac("%s  N_tot vs eq.(18)" % k, r["N_tot"][i_age], 4e53 * r["E_B_star"] / 1e50,
              2.5, "eq.(18) assumes all of E_B* injected, only %.0f%% is by t_age"
              % (100 * (1 - (t_age / r["t_0"]) ** (1 - a))))

    # 10.  eq. (19) late-time slope  RM ~ t^-(6+alpha)/2
    late = (r["t"] > 2 * t_age)
    slope = np.polyfit(np.log(r["t"][late]), np.log(r["RM"][late]), 1)[0]
    check("%s  d lnRM/d lnt vs -(6+a)/2" % k, slope, -(6 + a) / 2, 0.12)

    # 11.  Fig. 3: the epoch at which RM crosses the observed 1.46e5 rad m^-2
    check("%s  t_age from RM crossing [yr]" % k, crossing(r["t"], r["RM"]) / yr,
          t_age / yr, 0.25)

# ------------------------------------------------------------------
# 12-13.  Fig. 4 top: spectra at t_age/3, t_age, 3 t_age
# ------------------------------------------------------------------
nu_spec = np.logspace(7, 13, 150)
spectra = {}
for k, r in runs.items():
    spectra[k] = [
        (ts, M.calc_L_nu_spectrum(r["gam"], r["du"], nu_spec, Ns, Bs, Rs))
        for (ts, Ns, Bs, Rs) in r["snaps"]
    ]
    # the paper only quotes these off the top panel of Fig. 4: a self absorption
    # break "around nu ~ 5 GHz" for models A and C, and peak L_nu of order 1e30
    _, L_age = spectra[k][1]
    check_fac("%s  SSA break at t_age [GHz]" % k, nu_spec[L_age.argmax()] / 1e9, 5.0, 2.0)
    check_fac("%s  peak L_nu at t_age" % k, L_age.max(), 1e30, 5.0)

# ------------------------------------------------------------------
# 14.  grid convergence
# ------------------------------------------------------------------
r_hi = run("A", n_gam=1000, n_t=6000)
check("A  RM convergence, 500/3000 -> 1000/6000",
      crossing(r_hi["t"], r_hi["RM"]) / crossing(runs["A"]["t"], runs["A"]["RM"]),
      1.0, 0.05)

# ------------------------------------------------------------------
# 15.  the eq. (7) sign switch: the printed -3 must break conservation
# ------------------------------------------------------------------
M.EXPANSION_SIGN = -1.0
r_bad = run("A", n_gam=200, n_t=600)
M.EXPANSION_SIGN = +1.0
i_age = np.argmin(np.abs(r_bad["t"] - r_bad["t_age"]))
N_inj = 5e50 * (1 - (12.4 / 0.2) ** (1 - 1.3)) / ((1 + SIGMA) * XI)
_results.append((r_bad["N_tot"][i_age] / N_inj > 1e6,
                 "eq.(7) with the printed -3 breaks conservation",
                 r_bad["N_tot"][i_age] / N_inj, None, "must be >> 1"))

# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------
print("\n%-46s %13s %13s" % ("check", "got", "expected"))
print("-" * 92)
for ok, name, got, want, note in _results:
    w = "%13.4g" % want if want is not None else "%13s" % "-"
    print("%-4s %-41s %13.4g %s  %s" % ("PASS" if ok else "FAIL", name, got, w, note))
n_fail = sum(1 for r in _results if not r[0])
print("-" * 92)
print("%d/%d passed" % (len(_results) - n_fail, len(_results)))

# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------
cols = {"A": "r", "B": "b", "C": "goldenrod"}

fig, ax = plt.subplots(figsize=(6.5, 5))
for k, r in runs.items():
    ax.loglog(r["t"] / yr, r["RM"], color=cols[k], label="model %s" % k)
    ax.plot(crossing(r["t"], r["RM"]) / yr, RM_OBS, "*", color=cols[k], ms=14)
ax.axhline(RM_OBS, ls="--", color="k", lw=1)
ax.set_xlabel(r"$t$ (yr)")
ax.set_ylabel(r"RM (rad m$^{-2}$)")
ax.set_ylim(1e1, 1e9)
ax.legend()
ax.set_title("Fig. 3: rotation measure, eq. (16)")
fig.tight_layout()
fig.savefig("fig3_RM.png", dpi=130)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 9))
styles = ["--", "-", ":"]
for k in ("A", "C"):
    for (ts, L), ls in zip(spectra[k], styles):
        ax1.loglog(nu_spec / 1e9, L, color=cols[k], ls=ls, lw=2,
                   label="model %s" % k if ls == "-" else None)
nu_d, L_d = M.calc_L_nu_obs(DATA_NU_OBS, DATA_F_OBS)
ax1.errorbar(nu_d / 1e9, L_d, yerr=0.1 * L_d, fmt="ko", label="FRB 121102", zorder=10)
ax1.set_xlim(1e-2, 1e4)
ax1.set_ylim(1e28, 1e31)
ax1.set_xlabel(r"$\nu$ (GHz)")
ax1.set_ylabel(r"$L_{\nu}$ (erg s$^{-1}$ Hz$^{-1}$)")
ax1.legend()
ax1.set_title(r"Fig. 4 top: spectra at $t_{\rm age}/3,\ t_{\rm age},\ 3t_{\rm age}$")

rC = runs["C"]
for j, (nu_lc, lab) in enumerate(zip((325e6, 1.4e9, 3e9), ("325 MHz", "1.4 GHz", "3 GHz"))):
    ax2.loglog(rC["t"] / yr, rC["L_nu"][j], lw=2, label=lab)
ax2.axvline(rC["t_age"] / yr, ls="--", color="k", lw=1)
ax2.set_xlim(1, 1e2)
ax2.set_ylim(1e27, 1e31)
ax2.set_xlabel(r"$t$ (yr)")
ax2.set_ylabel(r"$L_{\nu}$ (erg s$^{-1}$ Hz$^{-1}$)")
ax2.legend()
ax2.set_title("Fig. 4 bottom: light curves, model C")
fig.tight_layout()
fig.savefig("fig4_spectra_lc.png", dpi=130)
print("\nwrote fig3_RM.png and fig4_spectra_lc.png")
