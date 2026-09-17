# Magnetar Wind Nebula Model

Python implementation of the synchrotron self-absorbed (SSA) magnetized ion–electron
wind nebula model of
[Margalit & Metzger 2018, ApJL 868, L4](https://iopscience.iop.org/article/10.3847/2041-8213/aaedad)
([arXiv:1808.09969](https://arxiv.org/abs/1808.09969)).

A flaring magnetar injects particles and magnetic energy into a freely expanding
nebula. The code evolves the electron distribution `N_gam(gam, t)` under adiabatic,
synchrotron (SSA-suppressed), inverse-Compton and bremsstrahlung cooling, and returns
the synchrotron spectrum, radio light curves and the rotation measure.

## Reproducing the paper

`test_MWN_model.py` runs the paper's models A, B and C and checks them against its
published results — 33/33 checks pass in ~30 s.

### Rotation measure (paper Fig. 3)

The age of the source is set by where the predicted RM falls through the observed
`1.46e5 rad m^-2` of FRB 121102 (Michilli et al. 2018, dashed line; stars mark the
crossings).

![RM vs time](fig3_RM.png)

| model | `E_B*` (erg) | `t_0` (yr) | `v_n` (cm/s) | `alpha` | `t_age` here | `t_age` paper |
|---|---|---|---|---|---|---|
| A | 5e50   | 0.2 | 3e8 | 1.30 | 14.7 yr | 12.4 yr |
| B | 5e50   | 0.6 | 1e8 | 1.30 | 44.5 yr | 37.8 yr |
| C | 4.9e51 | 0.2 | 9e8 | 1.83 | 15.1 yr | 13.1 yr |

Agreement is a consistent ~18%, well outside grid convergence (0.1%). The paper
states no numerical method, so ~20% is taken to be the achievable agreement.

### Spectra and light curves (paper Fig. 4)

Top: `L_nu` at `t_age/3` (dashed), `t_age` (solid) and `3 t_age` (dotted), against the
persistent source of FRB 121102 (Chatterjee et al. 2017). Bottom: model C light curves.

![Spectra and light curves](fig4_spectra_lc.png)

The self-absorption break sits at 5.5 GHz for model A and 2.6 GHz for model C (paper:
"around `nu ~ 5 GHz`"), with peak `L_nu` of 3.5e29–8.6e29 erg/s/Hz (paper: `~1e30`).
Models A and C reproduce the same tension with the lowest-frequency data point that
the paper itself reports.

## MCMC fit to the FRB 121102 data

`MWN_MCMC_fit.py` fits the model to the six persistent-source flux densities
(Chatterjee et al. 2017) **plus** the rotation measure (Michilli et al. 2018) — the RM
is what makes `t_age` identifiable, as in the paper. Four free parameters: `E_B*`,
`alpha`, `v_n`, `t_age`. `chi = 0.2 GeV`, `sigma = 0.1` and `t_0 = 0.2 yr` are fixed as
in the paper (`t_0` is not constrained by these data). The paper's own Fig. 1 limits
`t_age > 6 yr` and `2 R_n < 0.66 pc` are imposed as priors.

`MCMC_plotter.py` reports the best fit, the corner plot and the walks.

![Best fit spectrum](MWN_best_fit_spectrum.jpg)

64 walkers x 3028 steps, autocorrelation time 51–70.

![Corner plot](best_fit_corner_plot_MWN.jpg)

The tight `log10(v_n)`–`log10(t_age)` anticorrelation is the degeneracy noted below:
only their product `R_n` is measured. Dashed crosshairs mark the MAP.

![Walks](walks_MWN.jpg)

| parameter | marginal | MAP | model A | model B | model C |
|---|---|---|---|---|---|
| `log10(E_B*/erg)` | 50.82 (+0.10, −0.07) | 50.88 | 50.70 | 50.70 | 51.69 |
| `alpha`           | 1.71 (+0.12, −0.22)  | 1.83  | 1.30  | 1.30  | 1.83  |
| `log10(v_n)`      | 8.88 (+0.08, −0.13)  | 8.94  | 8.48  | 8.00  | 8.95  |
| `log10(t_age/yr)` | 0.81 (+0.13, −0.03)  | 0.80  | 1.09  | 1.58  | 1.12  |
| **`R_n = v_n t_age`** | **1.66e17 cm (±11%)** | 1.72e17 | 1.17e17 | 1.19e17 | 3.72e17 |

Quote the marginal column as the constraint; the MAP is the actual best-fitting sample
and is what the curves above are drawn at. **These differ because the parameters are
correlated** — the data constrain the product `R_n = v_n t_age`, so stacking the
independent 1-D marginals gives a vector that lies off the degeneracy ridge (its
`lnprob` is at the 1.3rd percentile of the chain). Plotting that stacked vector puts
the "best fit" curve outside its own credible band.

The fit lands on model A's energy scale and model C's `alpha` and `v_n`, with a nebula
radius between the two. `R_n` is the best-determined quantity — consistent with the
paper's analytic argument (eqs. 17–22) that `E_B*` and `R_n` are what the RM and the
self-absorption constraint actually pin down. At the MAP the RM comes out
1.438e5 rad m^-2 against the observed 1.46e5 (0.15 sigma).

Two honest caveats:

- **Reduced chi^2 = 16** (chi^2 = 48.6, 3 dof). 79% of it comes from the two lowest
  frequencies: 1.63 GHz (model 133 vs 250 uJy) and 3 GHz (290 vs 206); every other
  point is within 2 sigma and the RM within 0.2 sigma. This is the same failure the
  paper reports — the one-zone model's self-absorption turnover is sharper than the
  observed spectrum. The assumed 10% errors are also optimistic for a source known to
  vary.
- **`t_age` rails against the 6 yr prior floor** (24% of samples within 0.05 dex of it).
  The spectrum and RM on their own prefer a younger, faster-expanding nebula than the
  paper's 12.4 yr; the 6 yr limit comes from the source having been active since
  discovery, not from these data.

## Usage

```bash
python MWN_model.py        # model A demo run + plot
python test_MWN_model.py   # 33 checks against the paper, writes both figures above
python MWN_MCMC_fit.py     # MCMC fit  (add "--restart N" to extend an existing chain)
python MCMC_plotter.py     # best fit params, corner plot, walks, spectrum
```

```python
import numpy as np
import MWN_model as M
from MWN_model import yr

gam, du, dgam = M.calc_gam_grid(1.0, 1e5, 500)
t = np.logspace(np.log10(0.2 * yr), np.log10(43 * yr), 3000)   # seconds

snaps, RM, L_nu, N_tot, B_n, R_n = M.calc_L_nu_N_gam_t(
    t, gam, du, dgam, nu_arr=[3e9], t_0=0.2 * yr, alpha=1.3, B_16=1.0,
    xi=3.204e-4, xi_min=3.204e-4, v_n=3e8, sigma=0.1,
    E_B_star=5e50, snap_t=[12.4 * yr],
)
t_snap, N_gam, B, R = snaps[0]
L = M.calc_L_nu_spectrum(gam, du, np.logspace(7, 13, 150), N_gam, B, R)
```

Everything is cgs and **every time is in seconds**.

## Code layout

One `calc_*` function per paper equation; equation numbers are in the docstrings.

| | |
|---|---|
| eqs. (1), (4), (5) | `calc_E_dot`, `calc_N_e_dot` |
| eq. (6) | `calc_N_gam_0`, `calc_N_gam_inj` |
| eq. (7) | `calc_L_nu_N_gam_t` (the solver) |
| eqs. (8)–(10) | `calc_gamma_dot_adiab`, `calc_gamma_dot_syn_IC`, `calc_gamma_dot_brem`, `calc_L_rad` |
| eq. (11) | `calc_Rn`, `calc_EB_and_Bn` |
| eqs. (12)–(14) | `calc_L_nu`, `calc_j_nu`, `calc_alpha_nu`, `calc_P_nu`, `calc_nu_c`, `calc_F`, `calc_L_nu_spectrum` |
| eq. (15) | `calc_K_matrix`, `calc_alpha_nu_at_nu_c`, `calc_tau_gamma`, `calc_fssa` |
| eq. (16) | `calc_RM` |

Eq. (7) is stiff — synchrotron cooling near `t_0` would force `dt/t < 1e-7` explicitly
— so it is solved by backward Euler in `gam`. Every cooling term is negative, which
makes the implicit system bidiagonal and solvable in a single `solve_banded` call.
Expansion is applied as the exact factor `(t_prev/t)^3`, so particle number is
conserved to 1e-6. One model run takes ~5 s.

## Two deliberate departures from the printed paper

1. **Eq. (7) sign.** The paper prints `dN/dt + d(gdot N)/dgam - 3(Rdot_n/R_n) N_gam = N_gam_dot`.
   `N_gam` is a number *density*, so expansion must dilute it and the sign has to be
   `+3`. With the printed `-3` the nebula ends up with 3.8e9 times the electrons it was
   ever injected. Set `EXPANSION_SIGN = -1.0` to reproduce the printed version.
2. **Eq. (10) gets an extra `beta`.** As printed, bremsstrahlung stays finite at
   `gam = 1` and cools electrons off the bottom of the grid. The `beta` restores the
   `v/c` scaling so `gamma_dot -> 0` as `gam -> 1`.

## Known caveats

- The light-curve index `F_nu ~ t^-(alpha^2 + 7 alpha - 2)/4` does **not** hold at a
  fixed 3 GHz — measured −3.6 (A) and −6.2 (C) against the predicted −2.2 and −3.5.
  The RM index `-(6 + alpha)/2` does hold and is asserted by the tests.
- Eq. (19) is a scaling, not a normalisation: at model A / 12.4 yr it gives
  `1.1e6 rad m^-2`, 7.7x the observed value, because eq. (18) assumes all of `E_B*` has
  been injected (only 71% has by `t_age`) and that every electron sits at `gam = 1`.
- Eq. (17) for `B_n` is good to 13% for model A but 68% off for model C, since it drops
  the `1/(2 - alpha)` of eq. (11).

⚠️ Being adapted for SN 2004dk — the physics above is validated, the source-specific
parameters are not yet. ⚠️
