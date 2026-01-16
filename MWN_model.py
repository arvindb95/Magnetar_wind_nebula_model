import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import trapezoid, quad
from scipy.special import kv, kn
import astropy.units as u
import astropy.constants as const

# ==========================================
# 1. PHYSICAL CONSTANTS
# ==========================================
m_e = float(const.m_e.cgs.value)
c   = float(const.c.cgs.value)
e   = float(const.e.esu.value)
sigma_T = float(const.sigma_T.cgs.value)
alpha_fs = 1.0 / 137.036
yr_to_sec = 3.15576e7

# ==========================================
# 2. MODEL C PARAMETERS
# ==========================================
E_B_star = 4.9e51        # Total Magnetic Energy (erg)
t_0_yr   = 0.2           # Onset time (years)
alpha    = 1.83          # Braking index
v_n      = 9e8           # Expansion velocity (cm/s)
xi       = 0.1           # Efficiency
xi_erg   = 0.1 * u.GeV.to(u.erg) # Mean Particle Energy

# Frequencies to Plot
obs_freqs = [3.0e9, 1.4e9, 325e6]
freq_labels = ['3 GHz', '1.4 GHz', '325 MHz']
freq_colors = ['blue', 'green', 'orange']

# Grids
t_years = np.logspace(-2, 2, 100000)
gamma_grid = np.logspace(0, 6, 120)

# ==========================================
# 3. KERNELS
# ==========================================
def get_synchrotron_kernel(x):
    if x < 1e-5: return 2.15 * (x**(1/3))
    elif x > 50.0: return 0.0
    else:
        integrand = lambda k: kv(5.0/3.0, k)
        val, _ = quad(integrand, x, np.inf, limit=50)
        return x * val
F_x_exact = np.vectorize(get_synchrotron_kernel)

def maxwell_juttner_exact(gam, theta):
    gam = np.maximum(gam, 1.0001)
    beta = np.sqrt(1 - 1/gam**2)
    if theta < 1e-3: theta = 1e-3
    norm = theta * kn(2, 1.0/theta)
    if norm == 0: norm = 1.0
    return (gam**2 * beta * np.exp(-gam/theta)) / norm

# ==========================================
# 4. EVOLUTION LOOP
# ==========================================
t_sec = t_years * yr_to_sec
N_e = np.zeros(len(gamma_grid))
E_B = 1e42

L_storage = {nu: [] for nu in obs_freqs}
times_plotted = []
plot_stride = 500

print(f"Running Simulation with Bremsstrahlung...")

# Pre-calculate Matrix for SSA
gamma_i = gamma_grid[:, np.newaxis]
gamma_j = gamma_grid[np.newaxis, :]
x_matrix = (gamma_i / gamma_j)**2
Kernel_Matrix = F_x_exact(x_matrix)

for i in tqdm(range(1, len(t_sec))):
    t = t_sec[i]
    dt = t - t_sec[i-1]

    # --- Magnetar Engine ---
    term = (t_sec[i] / (t_0_yr * yr_to_sec))
    if term < 1: L_sd = (alpha - 1) * E_B_star / (t_0_yr * yr_to_sec)
    else: L_sd = (alpha - 1) * E_B_star / (t_0_yr * yr_to_sec) * (term**(-alpha))

    # --- Environment ---
    sigma_mag = 0.01
    source = L_sd * (sigma_mag / (1 + sigma_mag))
    loss = E_B / t
    E_B += (source - loss) * dt
    R = v_n * t
    Vol = (4.0/3.0) * np.pi * R**3
    B = np.sqrt(6 * E_B / Vol)

    # --- PHYSICS: SSA FEEDBACK ---
    n_dist = N_e / Vol
    phase_space_density = n_dist / (gamma_grid**2)
    df_dgam = np.gradient(phase_space_density, gamma_grid)

    nu_c_grid = (3 * e * B * gamma_grid**2) / (4 * np.pi * m_e * c)
    P_prefactor = (np.sqrt(3) * e**3 * B) / (m_e * c**2)
    integrand_vec = (gamma_grid**2) * df_dgam

    # Use trapezoid
    integral_result = trapezoid(Kernel_Matrix * integrand_vec[np.newaxis, :], gamma_grid, axis=1)
    alpha_vec = -1 * P_prefactor * integral_result / (8 * np.pi * m_e * nu_c_grid**2)

    tau_vec = np.maximum(alpha_vec * R, 1e-20)
    f_ssa = (1.0 - np.exp(-tau_vec)) / tau_vec

    # --- COOLING RATES ---
    g_dot_adiab = -gamma_grid / t

    U_B = B**2 / (8 * np.pi)
    const_synch = (4.0/3.0) * sigma_T * U_B / (m_e * c)
    g_dot_synch = -const_synch * (gamma_grid**2) * f_ssa

    # Bremsstrahlung
    N_total = trapezoid(N_e, gamma_grid)
    n_density_total = N_total / Vol
    g_dot_brem = -(5.0/3.0) * c * sigma_T * alpha_fs * n_density_total * np.sqrt(gamma_grid)

    g_dot = g_dot_adiab + g_dot_synch + g_dot_brem

    # --- Injection ---
    L_part = L_sd * (1.0 / (1 + sigma_mag)) * xi
    N_inj_rate = L_part / xi_erg
    theta_inj = (xi_erg / (m_e * c**2)) / 3.0
    shape = maxwell_juttner_exact(gamma_grid, theta_inj)
    norm_shape = trapezoid(shape, gamma_grid)
    if norm_shape > 0: shape /= norm_shape
    Q_inj = N_inj_rate * shape

    # --- Update ---
    flux = N_e * np.abs(g_dot); flux_in = np.roll(flux, -1); flux_in[-1] = 0
    dgamma = np.gradient(gamma_grid)
    N_e += ((flux_in - flux)/dgamma + Q_inj) * dt
    N_e = np.maximum(N_e, 0)

    # --- Radiation ---
    if i % plot_stride == 0:
        times_plotted.append(t / yr_to_sec)
        for nu_obs in obs_freqs:
            x_args = nu_obs / nu_c_grid
            F_values = F_x_exact(x_args)
            P_single = P_prefactor * F_values
            j_nu_4pi = trapezoid(N_e * P_single, gamma_grid)

            integrand_alpha = P_single * (gamma_grid**2) * df_dgam
            integral_alpha = trapezoid(integrand_alpha, gamma_grid)
            alpha_nu = -1 * integral_alpha / (8 * np.pi * m_e * nu_obs**2)

            if alpha_nu > 1e-30:
                tau = alpha_nu * R
                atten = (1.0 - np.exp(-tau)) / tau
            else:
                atten = 1.0
            L_storage[nu_obs].append((4 * np.pi * j_nu_4pi) * atten)

# ==========================================
# 5. FINAL PLOT
# ==========================================
times_plotted = np.array(times_plotted)
plt.figure(figsize=(10, 7))

for nu, color, label in zip(obs_freqs, freq_colors, freq_labels):
    L_data = np.array(L_storage[nu])
    mask = (L_data > 0) & np.isfinite(L_data)
    plt.loglog(times_plotted[mask], L_data[mask],
               linewidth=2.5, color=color, label=label)

if len(times_plotted) > 0:
    slope_exponent = (alpha**2 + 7*alpha - 2) / 4.0
    mask_guide = times_plotted > 30.0
    t_guide = times_plotted[mask_guide]

    if len(t_guide) > 0:
        L_anchor = L_storage[1.4e9][-1]
        t_anchor_pt = times_plotted[-1]
        L_guide = (L_anchor * 3) * (t_guide / t_anchor_pt)**(-slope_exponent)
        plt.loglog(t_guide, L_guide, 'r--', lw=2,
                   label=f'Late Time Slope $\propto t^{{-{slope_exponent:.2f}}}$')

plt.xlabel('Time [Years]', fontsize=14)
plt.ylabel(r'Luminosity $L_\nu$ [erg/s/Hz]', fontsize=14)
plt.title(f'Final Rigorous Model C\n(With SSA Feedback + Bremsstrahlung)', fontsize=16)
plt.grid(True, which='both', alpha=0.3)
plt.legend(fontsize=12)
plt.ylim(1e26, 1e36)
plt.xlim(0.1, 100)
plt.show()
