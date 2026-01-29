import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import trapezoid, quad
from scipy.special import kv, kn
import astropy.constants as const

# ============================================================
# 1. CONSTANTS
# ============================================================
m_e = const.m_e.cgs.value
c   = const.c.cgs.value
e   = const.e.esu.value
sigma_T = const.sigma_T.cgs.value
alpha_fs = 1/137.0
yr_to_sec = 3.15576e7
DL_Mpc = 972.0
DL_cm = DL_Mpc * 1e6 * const.pc.cgs.value
Area_Factor = 4 * np.pi * DL_cm**2

# Redshift
z_frb = 0.193

# ============================================================
# 2. EXACT KERNEL & PHYSICS FUNCTIONS
# ============================================================
gamma_grid = np.logspace(0, 7, 500) 

def get_synch_kernel_exact(x):
    val, _ = quad(lambda k: kv(5/3, k), x, np.inf, limit=100)
    return x * val

F_x_exact = np.vectorize(get_synch_kernel_exact)

gamma_i = gamma_grid[:, None]
gamma_j = gamma_grid[None, :]

def maxwell_juttner(gamma, theta):
    gamma = np.maximum(gamma, 1.0000001)
    beta = np.sqrt(1 - gamma**-2)
    norm = theta * kn(2, 1/theta)
    if norm == 0: norm = 1.0
    return gamma**2 * beta * np.exp(-gamma/theta) / norm

# ============================================================
# 3. FINITE VOLUME SOLVER
# ============================================================
def run_model(params, label):
    print(f"Running {label}...")
    E_B_star = params["E_B_star"]
    t0 = params["t_0_yr"] * yr_to_sec
    alpha = params["alpha"]
    v_n = params["v_n"]
    t_age_obs = params["t_age"] # This is the OBSERVED age (13.1 yr)
    
    # --- MODEL C PARAMETERS ---
    chi = 0.000320435  # ~0.2 GeV
    sigma_mag = 0.1

    t_age_src = t_age_obs # if we want to include redshift divide by(1 + z_frb)

    # Run simulation to Source Time 
    t_max = 3.2 * t_age_src 
    t_yr = np.logspace(-2, np.log10(t_max), 5000)
    t_sec = t_yr * yr_to_sec
    
    N_e = np.zeros_like(gamma_grid)
    E_B = 1e42 
    
    snapshots = {}
    
    # Convert targets to Source Frame for consistent checking
    targets_obs = {"Early": t_age_obs/3, "Mid": t_age_obs, "Late": 3*t_age_obs}
    targets_src = {k: v / (1 + z_frb) for k, v in targets_obs.items()}
    
    target_keys = sorted(targets_src.keys(), key=lambda k: targets_src[k])
    target_idx = 0

    print("  Computing Exact Matrix ...")
    X_mat = (gamma_i / gamma_j)**2
    Kernel_Matrix = F_x_exact(X_mat)
    print("  Done.")

    f_ssa_grid = np.ones_like(gamma_grid)

    for i in tqdm(range(1, len(t_sec))):
        t = t_sec[i]
        dt = t - t_sec[i-1]
        t_yr_now = t / yr_to_sec
        
        # 1. Dynamics
        if t < t0: L_sd = (alpha-1)*E_B_star/t0 
        else:      L_sd = (alpha-1)*E_B_star/t0*(t/t0)**(-alpha)
        
        expansion = 1.0/t
        loss_B = expansion * E_B
        source_B = L_sd * sigma_mag/(1+sigma_mag)
        E_B += (source_B - loss_B)*dt
        
        R = v_n * t
        Vol = 4.0/3.0 * np.pi * R**3
        B = np.sqrt(8*np.pi*E_B/Vol)
        U_B = B**2/(8*np.pi)
        
        # f_ssa Calculation, every 20 steps
        if i % 20 == 0:
            P_pref_const = (2.0/np.sqrt(3)) * e**3 * B / (m_e * c**2)
            nu_c_grid = (e * B * gamma_grid**2) / (2 * np.pi * m_e * c) 
            
            n_dist = N_e / Vol
            dfdg = np.gradient(n_dist / gamma_grid**2, gamma_grid)
            
            integrand = Kernel_Matrix * (gamma_grid**2 * dfdg)[None, :]
            integral = trapezoid(integrand, gamma_grid, axis=1)
            
            # Alpha at nu_c
            alpha_at_nuc = -(P_pref_const * integral) / (8 * np.pi * m_e * nu_c_grid**2)
            tau_at_nuc = np.maximum(alpha_at_nuc * R, 1e-30)
            f_ssa_grid = (1.0 - np.exp(-tau_at_nuc)) / tau_at_nuc

        # 2. Cooling
        gdot_ad = -gamma_grid * (1 - gamma_grid**-2) * expansion
        gdot_syn = -(4.0/3.0)*sigma_T*(U_B/(m_e*c))*(1 - gamma_grid**-2)*gamma_grid**2 * f_ssa_grid
        
        n_tot = trapezoid(N_e, gamma_grid)/Vol
        gdot_brem = -(5.0/3.0)*c*sigma_T*alpha_fs*n_tot*(gamma_grid)**1.2

        gdot = gdot_ad + gdot_syn + gdot_brem 
        
        # 3. Injection (No expansion term)
        theta = (chi/(m_e*c**2))/6.0 
        inj_shape = maxwell_juttner(gamma_grid, theta)
        norm = trapezoid(inj_shape, gamma_grid)
        if norm > 0: inj_shape /= norm
        Q_inj = (L_sd/((1+sigma_mag)*chi)) * inj_shape

        S = N_e + Q_inj * dt # By acknowledging it is a typo we have excluded an expansion term
        
        # 4. Implicit Update
        flux_from_above = 0.0 
        for j in range(len(gamma_grid)-1, -1, -1):
            if j == len(gamma_grid)-1: dg = gamma_grid[j] - gamma_grid[j-1]
            elif j == 0:               dg = gamma_grid[1] - gamma_grid[0]
            else:                      dg = 0.5 * (gamma_grid[j+1] - gamma_grid[j-1])
            
            cool_rate = np.abs(gdot[j]) * dt / dg
            numerator = S[j] + (flux_from_above * dt / dg)
            denominator = 1.0 + cool_rate
            N_e[j] = numerator / denominator
            flux_from_above = N_e[j] * np.abs(gdot[j])
            
        # Snapshot check against Source Time targets
        if target_idx < 3:
            current_target = targets_src[target_keys[target_idx]]
            if t_yr_now >= current_target:
                snapshots[target_keys[target_idx]] = {
                    "N_e": N_e.copy(), "B": B, "R": R, "Vol": Vol
                }
                target_idx += 1
                
    return snapshots

# ============================================================
# 4. SPECTRUM
# ============================================================
def spectrum(state):
    nu_obs_grid = np.logspace(8, 12, 100)
    N_e, B, R, Vol = state["N_e"], state["B"], state["R"], state["Vol"]
    
    nu_emit_grid = nu_obs_grid # if we want to include redshift multiply by (1 + z_frb)

    nu_c = (e*B*gamma_grid**2)/(2*np.pi*m_e*c)
    P_pref = (2.0/np.sqrt(3))*e**3*B/(m_e*c**2)
    
    n_dist = N_e/Vol
    dfdg = np.gradient(n_dist/gamma_grid**2, gamma_grid)
    
    Lnu_obs_list = []
    
    for nu_em in nu_emit_grid:
        x = nu_em/nu_c
        P = P_pref * F_x_exact(x)
        
        L_thin_rest = trapezoid(N_e * P, gamma_grid)
        alpha_rest = -trapezoid(P * (gamma_grid**2) * dfdg, gamma_grid) / (8*np.pi*m_e*nu_em**2)
        
        tau = alpha_rest * R
        atten = (1.0 - np.exp(-tau)) / tau
        L_em_atten = L_thin_rest * atten
        
        # Luminosity K-Correction (Specific Luminosity)
        L_obs = L_em_atten * (1 + z_frb) #redshift included
        Lnu_obs_list.append(L_obs)
        
    return nu_obs_grid, np.array(Lnu_obs_list)

# ============================================================
# 5. EXECUTION
# ============================================================
params_A = dict(E_B_star=5e50, t_0_yr=0.2, alpha=1.3, v_n=3e8, t_age=12.4)
params_C = dict(E_B_star=4.9e51, t_0_yr=0.2, alpha=1.83, v_n=9e8, t_age=13.1)

snaps_A = run_model(params_A, "Model A")
snaps_C = run_model(params_C, "Model C")

plt.figure(figsize=(10, 8))

data_nu = np.array([1.63, 3.0, 6.0, 10.0, 15.0, 22.0]) * 1e9
data_flux_uJy = np.array([250, 206, 203, 166, 103, 66])
data_L = data_flux_uJy * 1e-29 * Area_Factor 

plt.errorbar(data_nu, data_L, yerr=data_L*0.1, fmt='ko', label='FRB 121102', zorder=10)

styles = {"Early":"--", "Mid":"-", "Late":":"}
for k in ["Early", "Mid", "Late"]:
    if k in snaps_A:
        nu, L = spectrum(snaps_A[k])
        lbl = f"Model A" if k=="Mid" else None
        plt.loglog(nu, L, "r", linestyle=styles[k], linewidth=2, label=lbl)

for k in ["Early", "Mid", "Late"]:
    if k in snaps_C:
        nu, L = spectrum(snaps_C[k])
        lbl = f"Model C" if k=="Mid" else None
        plt.loglog(nu, L, color="goldenrod", linestyle=styles[k], linewidth=2, label=lbl)

plt.xlabel(r"Observed Frequency $\nu$ [Hz]", fontsize=12)
plt.ylabel(r"Luminosity $L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]", fontsize=12)
plt.title(r"Nebula Spectrum", fontsize=14)
plt.xlim(1e8, 1e12)
plt.ylim(1e27, 1e31)
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
