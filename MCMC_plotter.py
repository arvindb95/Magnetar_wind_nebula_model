import emcee
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from chainconsumer import (
    Chain,
    ChainConfig,
    ChainConsumer,
    PlotConfig,
    Truth,
    make_sample,
    truth,
)
import pandas as pd
from astropy.table import Table
from tqdm import tqdm

from MWN_MCMC_fit import (
    MWN_flux_density,
    fixed_params,
    freqs,
    fluxes,
    fluxerrs,
    RM_obs,
    RM_err,
    var_params_names,
    param_scale_log,
)

mpl.rcParams["font.size"] = 13
mpl.rcParams["legend.fontsize"] = 10

filename = "./FRB121102_MWN_fit.h5"
reader = emcee.backends.HDFBackend(filename)

tau = reader.get_autocorr_time(tol=0)
print(tau)
burnin = int(2 * np.max(tau))

samples = reader.get_chain(discard=burnin, flat=True)
log_prob = reader.get_log_prob(discard=burnin, flat=True)
print(np.shape(samples))

# Maximum likelihood sample.  The per-parameter marginal centres below are the right
# thing to QUOTE as constraints, but stacking them into a vector does not give a point
# of the posterior when the parameters are correlated (here alpha, v_n and t_age are,
# because the data constrain the product R_n = v_n t_age).  That stacked vector lands
# off the degeneracy ridge and its spectrum falls outside the credible band, so the
# curves and the chi^2 below use the MAP sample instead.
map_values = samples[np.argmax(log_prob)]

ndim = 4

labels = [
    r"$\rm{{log}}_{{10}}(E_{{B\star}})$",
    r"$\alpha$",
    r"$\rm{{log}}_{{10}}(v_{{n}})$",
    r"$\rm{{log}}_{{10}}(t_{{age}})$",
]

tab_labels = ["log_E_B_star", "alpha", "log_v_n", "log_t_age"]

df = pd.DataFrame(samples, columns=labels)

c = ChainConsumer()
c.add_chain(Chain(samples=df, name="data table"))


best_fit_values = np.zeros(len(labels))
ll_values = np.zeros(len(labels))
ul_values = np.zeros(len(labels))

loc_best_fit = {}

for i in range(len(labels)):
    param_best_fit = c.analysis.get_summary()["data table"][labels[i]].center
    param_ll = c.analysis.get_summary()["data table"][labels[i]].lower
    param_ul = c.analysis.get_summary()["data table"][labels[i]].upper

    best_fit_values[i] = param_best_fit
    ll_values[i] = param_ll
    ul_values[i] = param_ul
    loc_best_fit.update({labels[i]: param_best_fit})


mcmc_best_fit_tab = Table(
    [
        tab_labels,
        best_fit_values,
        ll_values,
        ul_values,
        map_values,
    ],
    names=["param", "best_fit", "ll", "ul", "map"],
)


mcmc_best_fit_tab.write("mcmc_best_fit_params_MWN.txt", format="ascii", overwrite=True)

## ------------ Report the best fit ------------ ##


def params_from_theta(theta):
    """
    Turns a vector of sampled parameters into the kwargs MWN_flux_density wants.
    """
    out = {}
    for i in range(len(tab_labels)):
        out.update(
            {var_params_names[i]: 10 ** theta[i] if param_scale_log[i] else theta[i]}
        )
    return out


best_fit_params = params_from_theta(map_values)

model_flux, model_RM = MWN_flux_density(freqs, **best_fit_params, **fixed_params)

# chi^2 over the 6 flux densities plus the rotation measure
n_of_obs = len(fluxes) + 1
n_of_fit_params = ndim
dof = n_of_obs - n_of_fit_params


def calc_chi_sq(flux, flux_errs, model):
    return np.sum(((flux - model) ** 2) / (flux_errs**2))


chi_sq = calc_chi_sq(fluxes, fluxerrs, model_flux)
chi_sq += calc_chi_sq(RM_obs, RM_err, model_RM)

print("###############################")
print("%-14s   %-24s %s" % ("", "marginal (quote these)", "MAP (used for the curves)"))
for i in range(len(tab_labels)):
    print(
        "%-14s = %9.4f  (-%.4f, +%.4f)   %9.4f"
        % (
            tab_labels[i],
            best_fit_values[i],
            best_fit_values[i] - ll_values[i],
            ul_values[i] - best_fit_values[i],
            map_values[i],
        )
    )
print("-------------------------------")
for name, val in best_fit_params.items():
    print("%-14s = %.4g" % (name, val))
print("-------------------------------")
print("RM model = %.4e rad m^-2  (observed %.4e)" % (model_RM, RM_obs))
print("F_nu model (uJy) = ", np.array2string(model_flux, precision=1))
print("F_nu data  (uJy) = ", np.array2string(fluxes, precision=1))
print("Degrees of freedom = ", dof)
print("Chi_sq = ", chi_sq)
print("Reduced chi_sq = ", chi_sq / dof)
print("###############################")

# Now plotting

c.set_plot_config(
    PlotConfig(
        usetex=True,
        label_font_size=20,
        tick_font_size=15,
        contour_label_font_size=20,
        summary_font_size=18,
        sigma2d=True,
    )
)


loc_map = {labels[i]: map_values[i] for i in range(len(labels))}
c.add_truth(Truth(location=loc_map))  # crosshairs at the MAP, titles quote the marginals

c.plotter.plot()
plt.savefig("best_fit_corner_plot_MWN.jpg", dpi=300)

c.plotter.plot_walks(convolve=100, plot_weights=False)
plt.savefig("walks_MWN.jpg", dpi=300)

## ------------ Best fit spectrum against the data ------------ ##

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

fig = plt.figure()
ax = fig.add_subplot(111)

ax.errorbar(
    freqs,
    fluxes,
    yerr=fluxerrs,
    capsize=2,
    elinewidth=0.5,
    c="k",
    fmt=".",
    label=r"FRB 121102 (Chatterjee et al. 2017)",
)

nu_range = np.logspace(np.log10(0.3), np.log10(60), 60)

# 1 sigma spread from a random subset of the chain
n_draw = 400
draws = samples[np.random.randint(len(samples), size=n_draw)]
flux_draws = np.zeros((n_draw, len(nu_range)))

print("Calculating errors on F_nu")

for p in tqdm(range(n_draw)):
    flux_draws[p], _ = MWN_flux_density(
        nu_range, **params_from_theta(draws[p]), **fixed_params
    )

center_values = np.percentile(flux_draws, [16, 50, 84], axis=0)

ax.fill_between(
    nu_range,
    y1=center_values[0],
    y2=center_values[2],
    color="C0",
    alpha=0.3,
    zorder=-1,
    label=r"posterior $1\sigma$",
)
# the MAP is an actual sample, so this curve lies inside the band above
ax.plot(
    nu_range,
    MWN_flux_density(nu_range, **best_fit_params, **fixed_params)[0],
    linewidth=1.0,
    color="C0",
    zorder=10,
    label="MWN best fit (MAP)",
)

ax.legend()
ax.set_xlabel(r"Observed frequency (GHz)")
ax.set_ylabel(r"Flux density ($\mu$Jy)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(10, 1e3)

plt.savefig("MWN_best_fit_spectrum.jpg", dpi=300)
plt.show()
