import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kn, kv
from scipy.integrate import quad, trapezoid
import astropy.units as u
import astropy.constants as const

# ==========================================
# 1. PHYSICAL CONSTANTS (CGS)
# ==========================================
m_e = const.m_e.cgs.value
c = const.c.cgs.value
e = const.e.esu.value
sigma_T = const.sigma_T.cgs.value
yr_to_sec = 3.154e7

# ==========================================
# 2. PHASE 1: THE ENGINE (Magnetar Inputs)
# ==========================================
def calc_E_dot(t, t_0, alpha, B_16):
    E_b_star = 3e49 * (B_16**2) 
    term = (alpha - 1) * (E_b_star / (t_0 * yr_to_sec)) * ((t / t_0) ** (-alpha))
    if isinstance(t, np.ndarray): term[t < t_0] = 0
    elif t < t_0: return 0
    return term

def calc_N_e_dot(t, t_0, alpha, B_16, xi, sigma=0.1):
    E_dot = calc_E_dot(t, t_0, alpha, B_16)
    return E_dot / (1 + sigma) / xi

def calc_N_gam_0(gam, xi, xi_min):
    theta = (196 / 3) * (xi / xi_min)
    gam = np.maximum(gam, 1.0001) 
    beta = np.sqrt(1 - (1 / (gam**2)))
    return ((gam**2) * beta * np.exp(-gam / theta)) / (theta * kn(2, (1 / theta)))

# ==========================================
# 3. PHASE 2: THE ENVIRONMENT (B-Field & R)
# ==========================================
def calc_EB_and_Bn(t_array_yr, t_0, alpha, B_16, v_n, sigma):
    t_sec = t_array_yr * yr_to_sec
    E_B = np.zeros_like(t_sec)
    E_B[0] = 1e40  # Initial seed energy
    
    E_dot_array = calc_E_dot(t_array_yr, t_0, alpha, B_16)
    source_term = (sigma / (1 + sigma)) * E_dot_array
    
    for i in range(1, len(t_sec)):
        dt = t_sec[i] - t_sec[i - 1]
        loss_rate = E_B[i - 1] / t_sec[i - 1]
        E_B[i] = E_B[i - 1] + (source_term[i - 1] - loss_rate) * dt

    R_n = v_n * t_sec
    B_n = np.sqrt(6 * E_B / np.maximum(R_n**3, 1.0))
    return B_n, R_n

# ==========================================
# 4. PHASE 3: EVOLUTION (The Solver)
# ==========================================
def evolve_electron_population(t_array_yr, gamma_grid, B_n_array, R_n_array, 
                               t_0, alpha, B_16, xi, xi_min):
    print("Running Phase 3: Electron Cooling Evolution...")
    t_sec = t_array_yr * yr_to_sec
    N_matrix = np.zeros((len(t_sec), len(gamma_grid)))
    d_gamma = np.gradient(gamma_grid)
    
    for i in range(1, len(t_sec)):
        dt = t_sec[i] - t_sec[i-1]
        N_old = N_matrix[i-1, :]
        
        # Cooling Rates
        g_dot = -(gamma_grid / t_sec[i]) - ((4.0/3.0) * sigma_T / (m_e * c)) * (B_n_array[i]**2 / (8*np.pi)) * (gamma_grid**2)
        
        # Injection
        N_dot_inj = calc_N_e_dot(t_array_yr[i], t_0, alpha, B_16, xi)
        shape = calc_N_gam_0(gamma_grid, xi, xi_min)
        shape /= np.trapz(shape, gamma_grid)
        
        # Advection (Upwind Scheme)
        flux = N_old * np.abs(g_dot)
        flux_in = np.roll(flux, -1) ; flux_in[-1] = 0 
        advection = (flux_in - flux) / d_gamma
        
        N_matrix[i, :] = np.maximum(N_old + (N_dot_inj * shape + advection) * dt, 0)
        
    return N_matrix

# ==========================================
# 5. PHASE 4: RADIATION (The Spectrum)
# ==========================================
def calc_F(x):
    return 1.78 * (x**0.297) * np.exp(-x)

def calc_j_and_alpha(nu, N_gamma, gamma_grid, B_n):
    nu_c = (3 * e * B_n * gamma_grid**2) / (4 * np.pi * m_e * c)
    P_nu = ((np.sqrt(3) * e**3 * B_n) / (m_e * c**2)) * calc_F(nu / nu_c)
    
    j_nu = np.trapz(N_gamma * P_nu, gamma_grid) / (4 * np.pi)
    
    df_dgam = np.gradient(N_gamma / gamma_grid**2, gamma_grid)
    alpha_nu = -1 * np.trapz((gamma_grid**2) * P_nu * df_dgam, gamma_grid) / (8 * np.pi * m_e * nu**2)
    
    return j_nu, alpha_nu

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
# Setup Parameters
t_years = np.logspace(-4, 2, 20000) 
gamma_grid = np.logspace(0, 4.7, 100) 
nu_obs = np.logspace(6, 14, 100) 

B_16, v_n, sigma = 1, 2e9, 0.1
t_0, alpha = 1.0, 1.3
xi = 1.0 * u.GeV.to(u.erg)
xi_min = 0.5 * u.GeV.to(u.erg)

# Run Simulation
B_n, R_n = calc_EB_and_Bn(t_years, t_0, alpha, B_16, v_n, sigma)
N_gam_history = evolve_electron_population(t_years, gamma_grid, B_n, R_n, t_0, alpha, B_16, xi, xi_min)

# Plot Results
plt.figure(figsize=(10, 7))
target_years = [1, 3, 10, 30]
colors = ['red', 'orange', 'green', 'blue']

for i, yr in enumerate(target_years):
    idx = np.argmin(np.abs(t_years - yr))
    current_N, current_B, current_R = N_gam_history[idx], B_n[idx], R_n[idx]
    
    L_nu_array = []
    for freq in nu_obs:
        j, alpha_val = calc_j_and_alpha(freq, current_N, gamma_grid, current_B)
        if alpha_val > 1e-30:
            tau = alpha_val * current_R
            L_nu = ((4/3) * np.pi * current_R**3) * (4 * np.pi * j) * ((1 - np.exp(-tau)) / tau)
        else:
            L_nu = ((4/3) * np.pi * current_R**3) * 4 * np.pi * j
        L_nu_array.append(L_nu)
    
    plt.loglog(nu_obs / 1e9, L_nu_array, lw=2, color=colors[i], label=f'{yr} yrs')

plt.xlabel(r'Frequency [GHz]', fontsize=14)
plt.ylabel(r'Luminosity $L_\nu$ [erg/s/Hz]', fontsize=14)
plt.title(r'Nebula Evolution', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", alpha=0.3)
plt.xlim(1e-3, 1e5)
plt.ylim(bottom=1e26)
plt.tight_layout()
plt.show()
