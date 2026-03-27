## single redshift
## correct survey volume and k binning
## no AP effect
## all combined into ONE power spectrum function
## reads in a params.ini file
## one circular array

## import packages
import numpy as np
import matplotlib.pyplot as plt
import camb
from camb import model
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
import os

## load parameters from params.ini
def load_parameters(ini_file='params.ini'):
    # Get the directory where this script lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, ini_file)

    params = {}
    with open(full_path, 'r') as f:
        for line in f:
            # Remove everything after the first '#' (inline comment)
            if '#' in line:
                line = line[:line.index('#')]
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue  # skip malformed lines
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            # Convert to number if possible
            try:
                params[key] = int(value)
            except ValueError:
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value
    return params

# Load parameters at module level (or inside a function)
PARAMS = load_parameters()

# Filter out any keys that might conflict with built‑ins (optional safety check)
safe_keys = [k for k in PARAMS if not k.startswith('_') and k not in dir(__builtins__)]
globals().update({k: PARAMS[k] for k in safe_keys})

##----------------------------------------
## set the fiducial cosmology
fiducial = {
    'ombh2' : ombh2,
    'omch2' : omch2,
    'tau' : tau,
    'As' : As,
    'ns' : ns,
    'H0' : H0,
    'alpha_s' : alpha_s
}

h = fiducial['H0'] / 100

fiducial_arr = np.array([v for v in fiducial.values()])

wavelength = 0.211 * (redshift + 1) #m
freq = (3e8 / wavelength) * 1e-6 #MHz
t_tot = t_tot * 3600 ## convert from hours to seconds

## design an array
D_half_lambda = wavelength/2
if D_min < D_half_lambda:
    print('Warning: D_min is smaller than half wavelength')

## conversions
def z_to_freq(z):
    lam = (1+z) * 0.211
    frequency = (3e8 / lam) * 1e-6
    return frequency ## MHz

def freq_to_z(nu):
    '''
    nu: MHz
    '''
    lam = 3e8 / (nu * 1e6)
    z = (lam / 0.211) - 1
    return z

## basic cosmology quantities
def get_basic_cosmology(fid_dict, z):
    pars=camb.set_params(ombh2= fid_dict["ombh2"], omch2= fid_dict["omch2"], 
                            tau = fid_dict["tau"], As=fid_dict["As"], 
                            nrun=fid_dict["alpha_s"], ns= fid_dict["ns"], 
                            H0=fid_dict["H0"], num_massive_neutrinos=1)
    results= camb.get_results(pars) 

    r_z = results.comoving_radial_distance(z)
    BAO_res = results.get_BAO([z], pars)
    D_A = BAO_res[:,2] ## unit: Mpc
    H_z = BAO_res[:,1] ## unit: km/s/Mpc

    return results, r_z, H_z[0]

results, r_z, H_z = get_basic_cosmology(fiducial, redshift)

## run CAMB
def get_HI_power_spectrum_interp(fid_dict, z, k_upper_lim=200):
    ## set the cosmology
    pars=camb.set_params(ombh2= fid_dict["ombh2"], omch2= fid_dict["omch2"], 
                            tau = fid_dict["tau"], As=fid_dict["As"], 
                            nrun=fid_dict["alpha_s"], ns= fid_dict["ns"], 
                            H0=fid_dict["H0"], num_massive_neutrinos=1)
    
    ## set 21cm parameters 
    pars.Do21cm  =True

    pars.Evolve_delta_xe =True # ionization fraction perturbations
    pars.Evolve_baryon_cs  = True # accurate baryon perturbations
    pars.WantCls =False
    pars.SourceTerms.use_21cm_mK = True # Use dimensionless rather than mK units
    pars.set_matter_power(kmax=k_upper_lim, redshifts=[z])

    # get transfer functions
    
    results= camb.get_results(pars)    
    trans = results.get_matter_transfer_data()

    k = trans.transfer_data[0,:,0]* results.Params.h
    primordial_PK = results.Params.scalar_power(k)

    mono_trans = trans.transfer_data[model.Transfer_monopole-1, :, 0] * k ** 2
    baryon_trans = trans.transfer_data[model.Transfer_b-1, :, 0] * k ** 2

    # get the moments of power spectrum Delta^2(k) in units of mK^2
    Pk_0_tr = primordial_PK * mono_trans ** 2
    Pk_2_tr = 2 * primordial_PK * mono_trans * baryon_trans
    Pk_4_tr = primordial_PK * baryon_trans ** 2

    factor = 2 * np.pi ** 2 / (k ** 3)

    # build the interpolator in unit of mK^2 Mpc^3
    Pk_mono_spline = CubicSpline(k, Pk_0_tr * factor)
    Pk_dipole_spline = CubicSpline(k, Pk_2_tr * factor)
    Pk_quad_spline = CubicSpline(k, Pk_4_tr * factor)  

    interp_lis = [Pk_mono_spline, Pk_dipole_spline, Pk_quad_spline]

    return interp_lis

def HI_power_spectrum_2D(k_perp, k_para, interp_lis):
    k = np.sqrt(k_perp ** 2 + k_para ** 2)
    mu = np.sqrt(1 - (k_perp / k) ** 2)

    Pk_0 = interp_lis[0]
    Pk_2 = interp_lis[1]
    Pk_4 = interp_lis[2]

    PS_no_noise = Pk_0(k) + Pk_2(k) * mu ** 2 + Pk_4(k) * mu ** 4

    return PS_no_noise ## mK^2 Mpc^3

## Baseline distribution
def nb_D_FarView(D, N_ant, D_min, D_max, D0, w):
    integrand = lambda D: 2 * np.pi * D * (D - D0) ** 2 * np.exp(- ((D- D0) / w) ** 2)

    int_res = quad(integrand, D_min, D_max)

    A = (N_ant * (N_ant - 1) / 2) / int_res[0]

    return A * (D - D0) ** 2 * np.exp(- ((D - D0) / w) ** 2)

## Circular Array
def baseline_pdf(d, D, D_min):
    """
    Probability density function f(d) of baseline length d.
    Units: m⁻¹ (probability per meter)
    ∫₀ᴰ f(d) dd = 1
    """
    d = np.asarray(d)
    pdf = np.zeros_like(d)
    mask = (d > D_min) & (d < D)
    if np.any(mask):
        x = d[mask] / D
        term = np.arccos(x) - x * np.sqrt(1 - x**2)
        pdf[mask] = (16 * d[mask]) / (np.pi * D**2) * term
    return pdf

def baseline_radial_number_density(d, N, D, D_min):
    """
    Expected number of baselines per meter at length d.
    Units: m⁻¹
    ∫ dN/dd dd = total baselines = N(N-1)/2
    """
    total_baselines = N * (N - 1) / 2
    return total_baselines * baseline_pdf(d, D, D_min)

def nb_circular(d, N, D, D_min):
    """
    Expected number of baselines per square meter in the uv‑plane
    as a function of radial distance d.
    Units: m⁻²
    ∬ n(d) dA = ∫₀ᴰ n(d) · 2πd dd = total baselines
    """
    radial_density = baseline_radial_number_density(d, N, D, D_min)
    # Avoid division by zero at d=0 – the density is finite there.
    with np.errstate(divide='ignore', invalid='ignore'):
        uv_density = radial_density / (2 * np.pi * d)
    uv_density[d == 0] = 0  # or use a limit, but zero is fine for plotting
    return uv_density

def noise_power_spectrum(k_perp):
    '''
    P_N(k) with units
    '''
    u_val = k_perp * r_z / (2 * np.pi)
    D_val = u_val * wavelength

    T_sys = 5000 * (freq / freq_ref) ** (-2.5) #K
    A_eff = wavelength ** 2 * gain / (4 * np.pi) # m^2

    noise_coeff_1 = (wavelength / 1000) * (1 + redshift) * r_z ** 2 / H_z
    # noise_coeff_1 = (0.211/1000) * r_z ** 2 / H_z[0]
                        ##unit: km Mpc^2 / (km/s/Mpc)
                        ##      = s Mpc^3
    noise_coeff_2 = (wavelength ** 2 / A_eff) ** 2 ##dimensionless
    noise_coeff_3 = (1 / (N_pol * t_tot)) * (S_area / FoV) ##s^-1
    noise_coeff = T_sys ** 2 * noise_coeff_1 * noise_coeff_2 * noise_coeff_3

    nb_D = nb_circular(D_val, N_antenna, D_max, D_min)
    nb_u = nb_D * wavelength ** 2

    P_noise = noise_coeff / nb_u ##K^2 Mpc^3

    return 1e6 * P_noise ## mK^2 Mpc^3

##----------------------------------------------------
## set up the bins in k_perp and k_parallel
k_perp_min = 2 * np.pi * D_min / (r_z * wavelength)
k_perp_max = 2 * np.pi * D_max / (r_z * wavelength)

def k_para_max():
    freq_21 = (3e8 / 0.211) * 1e-6 ## MHz
    return  (freq_21/(channel_width * 1e-3)) * 2 * np.pi * H_z / (3e5 * (1+redshift) ** 2)

def k_para_min():
    bandwidth = freq * bandwidth_frac
    survey_depth_low = results.comoving_radial_distance(freq_to_z(freq + 0.5 * bandwidth))
    survey_depth_high = results.comoving_radial_distance(freq_to_z(freq - 0.5 * bandwidth))

    delta_r = survey_depth_high - survey_depth_low
    return 2 * np.pi / delta_r

k_parallel_max = k_para_max()
k_parallel_min = k_para_min()

## k_perp and k_para values in 1D array
## use the value at bin center
k_para_all_edges = np.arange(k_parallel_min, k_parallel_max, dk_para)
k_para_vals = k_para_all_edges[:-1] + 0.5 *dk_para

k_perp_all_edges = 10 ** np.arange(np.log10(k_perp_min), np.log10(k_perp_max), dlnk_perp)
dk_perp = np.array([k_perp_all_edges[kix+1] - k for kix, k in enumerate(k_perp_all_edges[:-1])])
k_perp_vals = k_perp_all_edges[:-1] + 0.5 * dk_perp

k_perp_grid, k_para_grid = np.meshgrid(k_perp_vals, k_para_vals[::-1])

k_grid = np.sqrt(k_perp_grid ** 2 + k_para_grid ** 2)
mu_grid = np.sqrt(1 - (k_perp_grid / k_grid) ** 2)

## interpolators for P(k) using fiducial cosmology
interp_lis_fid = get_HI_power_spectrum_interp(fiducial, redshift)

##--------------------------------------------------------
## survey volume and number of k-modes
def survey_volume():
    s_steradian = 0.000304617419786594 * S_area
    s_area_at_z = s_steradian * r_z ** 2 ## Mpc^2

    bandwidth = bandwidth_frac * freq
    delta_z = freq_to_z(freq - 0.5 * bandwidth) - freq_to_z(freq + 0.5 * bandwidth)

    survey_depth = np.absolute(results.comoving_radial_distance(redshift + delta_z) - r_z) ## Mpc
    
    v_survey = s_area_at_z * survey_depth ##Mpc^3

    return v_survey

def find_k_bin_width(kperp, kperp_edges):
    """
    Find the bin index and width for a given k_perp value.

    Parameters
    ----------
    kperp : float
        The value of k_perp to locate.
    kperp_edges : 1D numpy array
        Bin edges (as returned by generate_kperp_bins).

    Returns
    -------
    idx : int or None
        Index of the bin (0‑based) such that edges[idx] <= kperp < edges[idx+1].
        Returns None if kperp is outside the range [edges[0], edges[-1]).
    width : float or None
        Width of that bin (edges[idx+1] - edges[idx]).
        Returns None if kperp is outside the range.
    """
    # Use searchsorted to find the insertion point that maintains order.
    # The right side of the interval is open, so we use side='right' to get
    # the index i such that kperp < edges[i] (strictly). Then the bin is i-1.
    i = np.searchsorted(kperp_edges, kperp, side='right')
    idx = i - 1

    width = kperp_edges[idx+1] - kperp_edges[idx]
    return width

def number_k_modes(k_perp, k_para):
    V_survey = survey_volume() ## Mpc^3

    dk_perp_val = find_k_bin_width(k_perp, k_perp_all_edges)
    dk_para_val = find_k_bin_width(k_para, k_para_all_edges)
    d3k = 2 * np.pi * k_perp * dk_perp_val * dk_para_val

    Nk = d3k * V_survey / (2 * np.pi) ** 3

    return Nk

def delta_Pk(k_perp, k_para):
    '''
    total power spectrum error
    '''
    Nk = number_k_modes(k_perp, k_para)
    P_N = noise_power_spectrum(k_perp)
    Pk_21 = HI_power_spectrum_2D(k_perp, k_para, interp_lis_fid)

    delta_Pk = (1/np.sqrt(Nk)) * (P_N + Pk_21)

    return delta_Pk ## mK^2 Mpc^3

## 2D power spectrum error
deltaPK = delta_Pk(k_perp_grid, k_para_grid)
PS_HI_2D_fid = HI_power_spectrum_2D(k_perp_grid, k_para_grid, interp_lis_fid)

##-------------------------------------------
## Fisher matrix

##set the percentage for numerical derivative
##set a variation for alpha_s
percent = 0.05
alpha_s_vary = 0.05

def get_ps_vary_interp(params):
    fid_val = fiducial[params]
    if params == 'alpha_s':
        varied_val = alpha_s_vary
    else:
        varied_val = fid_val * (1 + percent)

    fisher = fiducial.copy()
    fisher[params] = varied_val

    interp_lis_vary = get_HI_power_spectrum_interp(fisher, redshift)

    return interp_lis_vary

## put the P(k, p_vary) interpolator into a dictionary
## each entry is a list of the three moments Pk0, Pk2, Pk4
Pk_interp_vary_dict = {}

for pix, param_name in enumerate(fiducial):
    interp_vary = get_ps_vary_interp(param_name)
    Pk_interp_vary_dict[param_name] = interp_vary


def fisher_element(param1, param2):
    ## param1
    interp_lis_vary_1 = Pk_interp_vary_dict[param1]
    PS_vary_1 = HI_power_spectrum_2D(k_perp_grid, k_para_grid, interp_lis_vary_1)

    ## param2
    interp_lis_vary_2 = Pk_interp_vary_dict[param2]
    PS_vary_2 = HI_power_spectrum_2D(k_perp_grid, k_para_grid, interp_lis_vary_2)

    dPk_1 = PS_vary_1 - PS_HI_2D_fid
    dPk_2 = PS_vary_2 - PS_HI_2D_fid

    if param1 == 'alpha_s':
        delta_p1 = alpha_s_vary
    else:
        delta_p1 = percent * fiducial[param1]

    if param2 == 'alpha_s':
        delta_p2 = alpha_s_vary
    else:
        delta_p2 = percent * fiducial[param2]

    dPk_dp_1 = dPk_1 / delta_p1
    dPk_dp_2 = dPk_2 / delta_p2

    delta_Pk_sqr = deltaPK ** 2

    integrand = dPk_dp_1 * dPk_dp_2 * (1 / delta_Pk_sqr)

    return np.sum(integrand)

# print(fisher_element('ombh2', 'ombh2'))

fiducial_fisher = fiducial.copy()
fiducial_fisher.pop('tau')

fisher_matrix = np.zeros((len(fiducial_fisher), len(fiducial_fisher)))

def run():
    for pix1, p1 in enumerate(fiducial_fisher):
        for pix2, p2 in enumerate(fiducial_fisher):
            entry = fisher_element(p1, p2)
            fisher_matrix[pix1, pix2] = entry

    cov = np.linalg.inv(fisher_matrix)

    for pix, params in enumerate(fiducial_fisher):
        err = np.sqrt(cov[pix, pix])
        print(params + ': ', err)

    ## calculate the number of independent modes 
    N_modes = np.sum((PS_HI_2D_fid/deltaPK)**2)

if __name__ == "__main__":
    run()
