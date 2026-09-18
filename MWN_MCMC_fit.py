## ---------------------- Import required packages ---------------------- ##

import os

# one BLAS thread per likelihood call; the Pool below does the parallelism
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import emcee
import astropy.units as u
import astropy.constants as const
from astropy.table import Table
from timeit import default_timer as timer
from MWN_model import *
import scipy.optimize as op
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib as mpl

params = {"font.family": "serif", "text.usetex": True}

mpl.rcParams.update(params)

## ---------------------- Begin Code ---------------------- ##

## Main physical parameters ##

# Fixed params
fixed_params = {
    "xi": 0.2 * u.GeV.to(u.erg),  # erg ; chi, mean energy per particle, eq. (5)
    "xi_min": 0.2 * u.GeV.to(u.erg),  # erg ; chi_min of eq. (6)
    "sigma": 0.1,  # magnetisation of the injected outflow
    "B_16": 1.0,  # unused, E_B_star is fitted directly instead of eq. (1)
    "t_0": 0.2,  # yr ; these data do not constrain it, so it is fixed as in the paper
    # grid: coarse enough for ~0.2 s per likelihood call, fine enough that the
    # model error (0.2% on RM, 3.5% on the fluxes) stays well inside the 10% data
    # errors.  kin_min is a floor on gam - 1, see calc_gam_grid.
    "kin_min": 1e-4,
    "gam_max": 1e5,
    "n_gam": 250,  # Lorentz factor grid points
    "n_t": 600,  # time steps from t_0 to t_age
    "z": 0.1927,  # redshift of FRB 121102
    "D_L_Mpc": 972.0,  # Mpc ; luminosity distance
}

# Variable parameters (starting guess is model A of the paper)

var_params = {
    "E_B_star": 5.0e50,  # erg
    "alpha": 1.3,
    "v_n": 3.0e8,  # cm/s
    "t_age": 12.4,  # yr
}

var_params_names = list(var_params.keys())
print(var_params_names)
param_scale_log = [True, False, True, True]

# alpha must satisfy 1 < alpha < 2 (eq. 4 needs alpha > 1, and the solution of
# eq. 11 is singular at alpha = 2).  The t_age floor of 6 yr is the paper's own
# lower limit from the source having been active since its discovery (Fig. 1).
bounds = [(49.0, 53.0), (1.05, 1.95), (7.5, 9.5), (np.log10(6.0), 2.0)]

# 2 R_n < 0.66 pc from VLBI imaging (Marcote et al. 2017), the other Fig. 1 limit
log_R_n_max = np.log10(0.5 * 0.66 * u.pc.to(u.cm))

guess_params = np.array(list(var_params.values()))
guess_params[param_scale_log] = np.log10(guess_params[param_scale_log])

## Physical constants ##

m_e = (const.m_e.cgs).value  # g
e = (const.e.esu).value  # esu
c = (const.c.cgs).value  # cm/s

## ------------ Load data ------------##

# Persistent radio source of FRB 121102 (Chatterjee et al. 2017), the black points
# in the top panel of Fig. 4 of the paper.  Errors are taken as 10 per cent.
comp_data = Table.read("./FRB121102_persistent_source_data.csv", format="ascii.csv")

freqs = comp_data["Freqs"].data
fluxes = comp_data["Fluxes"].data
fluxerrs = comp_data["Fluxerr"].data

# Rotation measure of FRB 121102 (Michilli et al. 2018, MJD 57747), the horizontal
# dashed line of Fig. 3.  This is what makes t_age identifiable.
RM_obs = 1.46e5  # rad m^-2
RM_err = 0.10 * RM_obs


## ------------ Define functions for MCMC ------------ ##


def MWN_flux_density(
    nu, E_B_star, t_0, alpha, v_n, t_age, xi, xi_min, sigma, B_16, kin_min, gam_max,
    n_gam, n_t, z, D_L_Mpc,
):
    """
    Evolves eq. (7) from t_0 to t_age for one parameter set and returns the flux
    density (uJy) at the observed frequencies nu (GHz) together with the rotation
    measure (rad m^-2), both at t_age.

    t_0 and t_age come in years, everything internal is cgs and seconds.
    """
    gam, du, dgam = calc_gam_grid(kin_min, gam_max, n_gam)
    t = np.logspace(np.log10(t_0 * yr), np.log10(t_age * yr), n_t)

    # nu_arr is left empty so that no light curve is built at every step -- only
    # the final state is needed here, which makes one likelihood call ~0.1 s
    snaps, RM, L_nu, N_tot, B_n, R_n = calc_L_nu_N_gam_t(
        t, gam, du, dgam, [], t_0 * yr, alpha, B_16, xi, xi_min, v_n, sigma,
        E_B_star=E_B_star, snap_t=[t[-1]],
    )
    t_snap, N_gam, B_snap, R_snap = snaps[0]

    # nu_emit = (1+z) nu_obs and F_nu = L_nu / (4 pi D_L^2 (1+z)), then erg -> uJy
    D_L = D_L_Mpc * 1e6 * (const.pc.cgs).value
    L_nu_model = calc_L_nu_spectrum(gam, du, (1 + z) * nu * 1e9, N_gam, B_snap, R_snap)

    return L_nu_model / (4 * np.pi * (D_L**2) * (1 + z)) / 1e-29, RM[-1]


def lnprior(theta):
    allow_param_set = True

    for param_id in range(len(theta)):
        allow_param_set = allow_param_set and (
            bounds[param_id][0] <= theta[param_id] <= bounds[param_id][1]
        )
    # VLBI upper limit on the nebula size, R_n = v_n t_age (Fig. 1 of the paper)
    allow_param_set = allow_param_set and (
        theta[2] + theta[3] + np.log10(yr) < log_R_n_max
    )

    if allow_param_set:
        return 0.0
    else:
        return -np.inf


def lnlike(theta, nu, F_obs, F_err):
    new_var_param_dict = {}

    for var_i in range(len(var_params_names)):
        if param_scale_log[var_i]:
            new_var_param_dict.update({var_params_names[var_i]: 10 ** (theta[var_i])})
        else:
            new_var_param_dict.update({var_params_names[var_i]: theta[var_i]})

    try:
        f_nu, RM = MWN_flux_density(nu, **new_var_param_dict, **fixed_params)
    except (ValueError, FloatingPointError):
        return -np.inf

    # eq. (12) is only meaningful while alpha_nu > 0; reject anything unphysical
    if not np.all(np.isfinite(f_nu)) or np.any(f_nu <= 0) or not np.isfinite(RM) or RM <= 0:
        return -np.inf

    inv_sigma2 = 1.0 / F_err**2.0
    inv_sigma2_RM = 1.0 / RM_err**2.0

    return -0.5 * (
        np.sum((F_obs - f_nu) ** 2 * inv_sigma2 - np.log(inv_sigma2))
        + (RM_obs - RM) ** 2 * inv_sigma2_RM
        - np.log(inv_sigma2_RM)
    )


def lnprob(theta, nu, F_obs, F_err):
    lp = lnprior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta, nu, F_obs, F_err)


def get_starting_pos(guess_parameters, nwalkers, ndim=4):
    pos = [
        np.asarray(list(guess_parameters)) + 1e-2 * np.random.randn(ndim)
        for i in range(nwalkers)
    ]

    return pos


def run_mcmc(
    guess_parameters,
    pool,
    backend_file,
    niters=500,
    nwalkers=200,
    ndim=4,
    restart=False,
):
    nu = freqs  ### Make sure this is in GHz, observed frame
    F = fluxes  ### Make sure this is in uJy
    F_err = fluxerrs  ### Make sure this is in uJy

    pos = get_starting_pos(guess_parameters, nwalkers, ndim=ndim)

    backend = emcee.backends.HDFBackend(backend_file)
    if restart == False:
        backend.reset(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            lnprob,
            args=(nu, F, F_err),
            backend=backend,
            pool=pool,
        )

        print("## ------------ Starting MCMC run ------------ ##")

        start = timer()
        sampler.run_mcmc(pos, niters, progress=True)
        end = timer()

        print("Computation time: %f s" % (end - start))

        tau = backend.get_autocorr_time(tol=0)
        print("The autocorrelation time for this run : ", tau)
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            lnprob,
            args=(nu, F, F_err),
            backend=backend,
            pool=pool,
        )

        print("Initial size of chain in backend file: {0}".format(backend.iteration))

        print("## ------------ Starting MCMC run ------------ ##")

        start = timer()
        sampler.run_mcmc(None, niters, progress=True)
        end = timer()

        print("Computation time: %f s" % (end - start))
        print("Final size of chain in backend file: {0}".format(backend.iteration))

        tau = backend.get_autocorr_time(tol=0)
        print("The autocorrelation time for this run : ", tau)

    return sampler


if __name__ == "__main__":
    import sys

    # "python MWN_MCMC_fit.py --restart 2700" continues an existing chain in the
    # backend file instead of starting a new one (and skips the minimization)
    restart = "--restart" in sys.argv
    niters = int(sys.argv[sys.argv.index("--restart") + 1]) if restart else 3000

    if restart:
        better_guess_params = {"x": guess_params}
    else:
        ## ------------ Initial minimization ------------ ##

        method = "Nelder-Mead"
        nll = lambda *args: -lnlike(*args)
        better_guess_params = op.minimize(
            nll,
            guess_params,
            bounds=bounds,
            args=(freqs, fluxes, fluxerrs),
            method=method,
            options={"disp": True, "maxiter": 2000},
        )
        print("The minimized parameters using " + method)
        print(better_guess_params["x"])

    with Pool() as pool:
        sampler = run_mcmc(
            better_guess_params["x"],
            niters=niters,
            nwalkers=64,
            ndim=4,
            pool=pool,
            backend_file="FRB121102_MWN_fit.h5",
            restart=restart,
        )
