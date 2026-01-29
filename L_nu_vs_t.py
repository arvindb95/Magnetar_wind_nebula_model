import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import trapezoid, quad
from scipy.special import kv, kn
import astropy.constants as const

# ==========================================
# 1. PHYSICAL CONSTANTS (CGS)
# ==========================================
m_e = const.m_e.cgs.value
c   = const.c.cgs.value
e   = const.e.esu.value
sigma_T = const.sigma_T.cgs.value
alpha_fs = 1/137.036
yr_to_sec = 3.15576e7
pc_to_cm = 3.0857e18

# ==========================================
# 2. MODEL C PARAMETERS
# ==========================================
# Source Properties
z_frb = 0.193
E_B_star = 4.9e51      # Total Energy (erg)
t_0_yr   = 0.2         # Onset time (yr)
alpha    = 1.83        # Braking index
v_n      = 9e8         # Expansion velocity (cm/s)

# Physics Parameters
chi_GeV = 0.2          # Energy per particle (GeV)
chi = chi_GeV * 1.60218e-3 # Converted to erg
sigma_mag = 0.1        # Magnetization

# Frequencies to Plot (Observer Frame)
obs_freqs = [3.0e9, 1.4e9, 0.325e9] # Hz
freq_labels = ['3 GHz', '1.4 GHz', '325 MHz']
colors = ['blue', 'green', 'orange']

# ==========================================
# 3. KERNELS & PHYSICS
# ==========================================
gamma_grid = np.logspace(0, 6.5, 300)

def get_synch_kernel_exact(x):
    if x > 20: return 0.0
    val, _ = quad(lambda k: kv(5/3, k), x, 100.0)
    return x * val

F_x_exact = np.vectorize(get_synch_kernel_exact)

# Pre-compute Matrix for SSA
gamma_i = gamma_grid[:, None]
gamma_j = gamma_grid[None, :]
Kernel_Matrix = F_x_exact((gamma_i / gamma_j)**2)

def maxwell_juttner(gamma, theta):
    gamma = np.maximum(gamma, 1.0000001)
    beta = np.sqrt(1 - gamma**-2)
    norm = theta * kn(2, 1/theta)
    if norm == 0: norm = 1.0
    return gamma**2 * beta * np.exp(-gamma/theta) / norm

# ==========================================
# 4. SIMULATION LOOP
# ==========================================
def run_simulation():
    print("Running Model C (L_nu vs Time)...")
    
    # Time Grid (SOURCE FRAME)
    t_max_obs = 100.0 
    t_max_src = t_max_obs #if we want to include redshift divide by (1 + z_frb)
    
    t_yr_src = np.logspace(-2, np.log10(t_max_src), 2000)
    t_sec_src = t_yr_src * yr_to_sec
    
    # Initialize State
    N_e = np.zeros_like(gamma_grid)
    E_B = 1e42 # Seed magnetic energy
    
    # Storage for Plotting
    results = {nu: [] for nu in obs_freqs}
    t_plot_list = [] # Will store OBSERVER times
    
    f_ssa_grid = np.ones_like(gamma_grid)

    for i in tqdm(range(1, len(t_sec_src))):
        t = t_sec_src[i]
        dt = t - t_sec_src[i-1]
        
        # A. Dynamics
        if t < t_0_yr * yr_to_sec: 
            L_sd = (alpha-1)*E_B_star/(t_0_yr*yr_to_sec)
        else:
            L_sd = (alpha-1)*E_B_star/(t_0_yr*yr_to_sec)*(t/(t_0_yr*yr_to_sec))**(-alpha)
            
        expansion_rate = 1.0/t
        
        # B-Field Evolution
        loss_B = expansion_rate * E_B
        source_B = L_sd * sigma_mag/(1+sigma_mag)
        E_B += (source_B - loss_B)*dt
        
        R = v_n * t
        Vol = 4.0/3.0 * np.pi * R**3
        B = np.sqrt(8*np.pi*E_B/Vol)
        U_B = B**2/(8*np.pi)
        
        # B. Calculate SSA Opacity (Every 10 steps)
        if i % 10 == 0:
            P_pref = (2.0/np.sqrt(3)) * e**3 * B / (m_e * c**2)
            nu_c_grid = (3 * e * B * gamma_grid**2) / (4 * np.pi * m_e * c)
            
            n_dist = N_e / Vol
            dfdg = np.gradient(n_dist / gamma_grid**2, gamma_grid)
            integrand = Kernel_Matrix * (gamma_grid**2 * dfdg)[None, :]
            integral = trapezoid(integrand, gamma_grid, axis=1)
            
            alpha_nu_grid = -(P_pref * integral) / (8 * np.pi * m_e * nu_c_grid**2)
            tau_grid = np.maximum(alpha_nu_grid * R, 1e-30)
            f_ssa_grid = (1.0 - np.exp(-tau_grid)) / tau_grid

        # C. Cooling (Expansion + Synchrotron + Bremsstrahlung) We have not included IC as it contributes only to the early onset times
        gdot_ad = -gamma_grid * (1 - gamma_grid**-2) * expansion_rate
        gdot_syn = -(4.0/3.0)*sigma_T*(U_B/(m_e*c))*gamma_grid**2 * f_ssa_grid
        
        n_tot = trapezoid(N_e, gamma_grid)/Vol
        gdot_brem = -(5.0/3.0)*c*sigma_T*alpha_fs*n_tot*(gamma_grid)**1.2
        
        gdot = gdot_ad + gdot_syn + gdot_brem
        
        # D. Injection
        theta = (chi/(m_e*c**2))/6.0 
        inj_shape = maxwell_juttner(gamma_grid, theta)
        norm = trapezoid(inj_shape, gamma_grid)
        if norm > 0: inj_shape /= norm
        Q_inj = (L_sd/((1+sigma_mag)*chi)) * inj_shape
        
        # E. Implicit Solve
        S = N_e + Q_inj * dt # By acknowledging it is a typo we have excluded an expansion term
        flux = 0.0
        for j in range(len(gamma_grid)-1, -1, -1):
            if j == len(gamma_grid)-1: dg = gamma_grid[j] - gamma_grid[j-1]
            elif j == 0:               dg = gamma_grid[1] - gamma_grid[0]
            else:                      dg = 0.5 * (gamma_grid[j+1] - gamma_grid[j-1])
            
            cool_rate = np.abs(gdot[j]) * dt / dg
            num = S[j] + (flux * dt / dg)
            den = 1.0 + cool_rate
            N_e[j] = num / den
            flux = N_e[j] * np.abs(gdot[j])
            
        # F. Calculate Light Curves (Every 20 steps)
        if i % 20 == 0:
            t_obs = t # if we want to include redshift, multiply by (1+z_frb)
            t_plot_list.append(t_obs / yr_to_sec)
            
            # Pre-compute spectral constants
            nu_c = (3 * e * B * gamma_grid**2) / (4 * np.pi * m_e * c)
            P_pref = (2.0/np.sqrt(3))*e**3*B/(m_e*c**2)
            n_dist = N_e/Vol
            dfdg = np.gradient(n_dist/gamma_grid**2, gamma_grid)
            
            for nu_obs in obs_freqs:
                nu_em = nu_obs # if we want to include redshift, multiply by (1+z_frb)   
                x = nu_em/nu_c
                P = P_pref * F_x_exact(x)
                
                # Emissivity
                j_nu_4pi = trapezoid(N_e * P, gamma_grid) # Total Power
                
                # Absorption
                integrand_alpha = P * (gamma_grid**2) * dfdg
                alpha_val = -trapezoid(integrand_alpha, gamma_grid) / (8*np.pi*m_e*nu_em**2)
                
                tau = alpha_val * R
                atten = (1.0 - np.exp(-tau)) / tau if tau > 1e-6 else 1.0
                
                L_val = j_nu_4pi * atten * (1+z_frb) #included redshift
                results[nu_obs].append(L_val)

    return np.array(t_plot_list), results

# ==========================================
# 5. EXECUTION & PLOTTING
# ==========================================
t_axis, L_data = run_simulation()

plt.figure(figsize=(10, 7))

# Plot Curves
for nu, color, label in zip(obs_freqs, colors, freq_labels):
    L = np.array(L_data[nu])
    plt.loglog(t_axis, L, color=color, linewidth=2.5, label=label)

# Formatting
plt.xlabel('Observer Time [Years]', fontsize=14)
plt.ylabel(r'Luminosity $L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]', fontsize=14)
plt.title('Model C Light Curve\n(Implicit Solver)', fontsize=14)
plt.xlim(1e-1, 100)
plt.ylim(1e27, 1e33)
plt.grid(True, which='both', alpha=0.3)
plt.legend(fontsize=12)

# Verify Slope (Late Time)
mask = t_axis > 20
if np.any(mask):
    t_fit = t_axis[mask]
    L_fit = np.array(L_data[1.4e9])[mask]
    # Simple power law guide that has been used in the paper
    plt.loglog(t_fit, L_fit[-1]*(t_fit/t_fit[-1])**(-4), 'k--', alpha=0.5, label=r'Slope $\sim t^{-4}$')

plt.tight_layout()
plt.show()
