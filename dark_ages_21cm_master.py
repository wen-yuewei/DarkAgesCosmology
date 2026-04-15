#!/usr/bin/env python3
"""
Forecast of power spectrum measurement and cosmological constraints
    based on the 21-cm power spectrum measured by a lunar far-side interferometer.

This code accompanies the paper:
    "A Designer’s Guide to Lunar Far-Side Interferometer Array: 
        Power Spectrum Measurement and Cosmological Constraints from the Dark Ages", 
        Yuewei Wen et al. (2026), Journal/arXiv:XXXX.XXXXX
"""

# ===================== import packages ========================
import numpy as np
import camb
from camb import model
from scipy.interpolate import CubicSpline
from scipy.integrate import quad, simpson
import os
import logging
import sys
from scipy.spatial.distance import pdist, squareform

logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)

# =============== load parameters from params.ini ==============
def load_parameters(ini_file='params.ini'):
    """
    Read parameters of about the array and survey from a .ini file.

    Parameters
    ----------
    ini_file : str, optional
        Path to the parameter file (default: 'params.ini').

    Returns
    -------
    dict
        Parameter names as keys, values converted to int/float where possible.
    """
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


## ================ set the fiducial cosmology ==================
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

## the redshifted 21-cm wavelength/frequency
wavelength = 0.211 * (redshift + 1) #m
freq = (3e8 / wavelength) * 1e-6 #MHz

def z_to_freq(z):
    """
    Convert redshift to frequency for the 21 cm line.

    Parameters
    ----------
    z : float
        Redshift.

    Returns
    -------
    float
        Frequency in MHz.
    """
    lam = (1+z) * 0.211
    frequency = (3e8 / lam) * 1e-6
    return frequency ## MHz

def freq_to_z(nu):
    """
    Convert frequency to redshift for the 21 cm line.

    Parameters
    ----------
    nu : float
        Frequency in MHz

    Returns
    -------
    float
        Redshift
    """
    lam = 3e8 / (nu * 1e6)
    z = (lam / 0.211) - 1
    return z

## process some survey parameters
D_half_lambda = wavelength/2
if D_min < D_half_lambda:
    logging.warning('D_min is smaller than half wavelength')

t_tot = t_tot * 3600 ## convert the total observation time from hours to seconds

# ===================== get basic cosmological quantities ===================
def get_basic_cosmology(fid_dict, z):
    """
    Compute basic cosmological quantities like radial comoving distance and Hubble parameter 
        for a given cosmology and redshift.

    Parameters
    ----------
    fid_dict : dict
        Dictionary of cosmological parameters (ombh2, omch2, tau, As, ns, H0, alpha_s).
    z : float
        Redshift.

    Returns
    -------
    results : CAMB results object
    r_z : float
        Comoving radial distance in Mpc.
    H_z : float
        Hubble parameter in km/s/Mpc.
    """
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

# Store these quantities for the AP effect 
D_A_tr = r_z / (1 + redshift) ## angular diameter distance in Mpc
Hz_tr = H_z

D_A_fid = D_A_tr
Hz_fid = Hz_tr

# ================= Calculate the 21-cm power spectrum wit CAMB ================
def get_HI_power_spectrum_interp(fid_dict, z, k_upper_lim=200):
    """
    Build cubic spline interpolators for the monopole, dipole, and quadrupole moments
        of the 21-cm power spectrum.

    Parameters
    ----------
    fid_dict : dict
        Dictionary of cosmological parameters (ombh2, omch2, tau, As, ns, H0, alpha_s).
    z : float
        Redshift.
    k_upper_lim : float, optional
        Maximum k for CAMB to compute (default: 200).

    Returns
    -------
    list of CubicSpline
        Interpolators for the monopole, dipole, and quadrupole moments (Pk0, Pk2, Pk4) 
            in units of mK^2 Mpc^3.
    """
    ## set the basic cosmology
    pars=camb.set_params(ombh2= fid_dict["ombh2"], omch2= fid_dict["omch2"], 
                            tau = fid_dict["tau"], As=fid_dict["As"], 
                            nrun=fid_dict["alpha_s"], ns= fid_dict["ns"], 
                            H0=fid_dict["H0"], num_massive_neutrinos=1)
    
    ## set 21cm-related parameters 
    pars.Do21cm = True
    pars.Evolve_delta_xe = True # ionization fraction perturbations
    pars.Evolve_baryon_cs = True # accurate baryon perturbations
    pars.WantCls = False
    pars.SourceTerms.use_21cm_mK = True # power spectrum with the mK units
    pars.set_matter_power(kmax=k_upper_lim, redshifts=[z])

    # get transfer functions
    results= camb.get_results(pars)    
    trans = results.get_matter_transfer_data()

    k = trans.transfer_data[0,:,0]* results.Params.h
    primordial_PK = results.Params.scalar_power(k) ## get the primordial power spectrum

    mono_trans = trans.transfer_data[model.Transfer_monopole-1, :, 0] * k ** 2 ## Eq. 2.2 in paper
    baryon_trans = trans.transfer_data[model.Transfer_b-1, :, 0] * k ** 2

    # get the three moments of power spectrum in units of mK^2
    Pk_0_tr = primordial_PK * mono_trans ** 2
    Pk_2_tr = 2 * primordial_PK * mono_trans * baryon_trans
    Pk_4_tr = primordial_PK * baryon_trans ** 2

    factor = 2 * np.pi ** 2 / (k ** 3) ## convert $\Delta^2(k)$ to P(k)

    # build the interpolator in unit of mK^2 Mpc^3
    Pk_mono_spline = CubicSpline(k, Pk_0_tr * factor)
    Pk_dipole_spline = CubicSpline(k, Pk_2_tr * factor)
    Pk_quad_spline = CubicSpline(k, Pk_4_tr * factor)  

    interp_lis = [Pk_mono_spline, Pk_dipole_spline, Pk_quad_spline]

    return interp_lis

## get interpolators for P(k) using the fiducial cosmology
interp_lis_fid = get_HI_power_spectrum_interp(fiducial, redshift)

def HI_power_spectrum_2D_tr(k_perp, k_para, interp_lis):
    """
    Compute the true 21-cm power spectrum without the Alcock-Paczynski effect.

    Parameters
    ----------
    k_perp : array_like
        Perpendicular wavenumber (in true cosmology).
    k_para : array_like
        Parallel wavenumber (in true cosmology).
    interp_lis : list of CubicSpline
        Interpolators generated from the `get_HI_power_spectrum_interp()` function.

    Returns
    -------
    array_like
        Power spectrum value at this (k_\perp, k_\parallel) in unit of mK^2 Mpc^3.
    """
    k = np.sqrt(k_perp ** 2 + k_para ** 2)
    mu = np.sqrt(1 - (k_perp / k) ** 2)

    Pk_0 = interp_lis[0]
    Pk_2 = interp_lis[1]
    Pk_4 = interp_lis[2]

    PS_no_noise = Pk_0(k) + Pk_2(k) * mu ** 2 + Pk_4(k) * mu ** 4

    return PS_no_noise ## mK^2 Mpc^3

def HI_power_spectrum_2D(k_perp_obs, k_para_obs, interp_lis_true,
                            D_A_ratio=1.0, Hz_ratio=1.0):
    """
    Compute the observed 21cm power spectrum including the Alock-Paczynski effect.

    Parameters
    ----------
    k_perp_obs : array_like
        Observed perpendicular wavenumber
    k_para_obs : array_like
        Observed parallel wavenumber
    interp_lis_true : list of CubicSpline
        Interpolators for the true power spectrum 
            generated from the `get_HI_power_spectrum_interp()` function.
    D_A_ratio : float
        Ratio D_A_fid / D_A_true
    Hz_ratio : float
        Ratio H_true / H_fid

    Returns
    -------
    array_like
        Observed power spectrum value at this (k_\perp, k_\parallel) in unit of mK^2 Mpc^3
    """
    # Map to true k values
    k_perp_true = D_A_ratio * k_perp_obs
    k_para_true = Hz_ratio * k_para_obs

    # Evaluate the true power spectrum at the mapped k values
    P_true = HI_power_spectrum_2D_tr(k_perp_true, k_para_true, interp_lis_true)

    # Volume scaling factor
    volume_factor = (D_A_ratio)**2 * Hz_ratio

    return P_true * volume_factor ## mK^2 Mpc^-3


# ===================== Baseline distributions =======================
def regular_polygon_distances(n, side):
    """
    Compute all distinct vertex‑to‑vertex distances for a regular n‑gon with side length `side`.
    Returns a list of distances and their multiplicities.

    Parameters
    ----------
    n : int
        Number of vertices of the regular n-gon
    side : float
        Side length of the regular n-gon

    Returns
    -------
    distances, multiplicities : tuple(list[float], list[int])
        First list contains distinct vertex-to-vertex distances
        Second list contains the corresponding multiplicities.   
    """
    if n < 2:
        return [], []  # no pairs
    # Circumradius

    R = side / (2 * np.sin(np.pi / n))
    distances = []
    multiplicities = []
    for k in range(1, n // 2 + 1):
        d = 2 * R * np.sin(np.pi * k / n)
        if k == n / 2 and n % 2 == 0:
            mult = n // 2
        else:
            mult = n
        distances.append(d)
        multiplicities.append(mult)
    return distances, multiplicities

def single_circular_pdf(d, D_min, D_max):
    """
    Probability density function w.r.t. baseline length for a single circular array.

    Parameters
    ----------
    d : array_like
        Baseline lengths in meters.
    D_min : float
        Minimum baseline length in meters.
    D_max : float
        Minimum baseline length in meters.

    Returns
    -------
    array_like
        radial probability for this given baseline length in unit of m^-1.
    """
    d = np.asarray(d)
    pdf = np.zeros_like(d)
    mask = (d > D_min) & (d < D_max)
    if np.any(mask):
        x = d[mask] / D_max
        term = np.arccos(x) - x * np.sqrt(1 - x**2)
        pdf[mask] = (16 * d[mask]) / (np.pi * D_max**2) * term
    return pdf

def f_cross_length(d_grid, L, D_min, D_max, n_theta=1000):
    """
    Probability distribution for cross-baselines.

    Parameters
    ----------
    d_grid : array_like
        Baseline lengths in meters
    L : float
        Separation between array centers in meters.
    D_min, D_max : float
        Minimum and maximum baseline lengths in meters.
    n_theta : int, optional
        Number of quadrature points for the numerical integration.

    Returns
    -------
    array_like
        Probability w.r.t. a cross-baseline of this length.
    """
    theta = np.linspace(0, 2*np.pi, n_theta)
    cos_theta = np.cos(theta).reshape(-1, 1, 1)
    d_reshaped = d_grid.reshape(1, *d_grid.shape)
    r = np.sqrt(d_reshaped**2 + L**2 - 2 * d_reshaped * L * cos_theta)
    pdf_vals = single_circular_pdf(r, D_min, D_max)
    integrand_vals = np.where(r < 1e-12, 0.0, pdf_vals / (2 * np.pi * r))
    integral = simpson(integrand_vals, x=theta, axis=0)
    return d_grid * integral 

# Read polygon parameters
n_stations = int(n_stations)          # from params.ini
side_len = float(L)         # from params.ini

# Precompute unique distances and their multiplicities
unique_Ls, mults = regular_polygon_distances(n_stations, side_len)

def nb_D_func(d, N, D_min, D_max):
    """
    Compute the baseline density distribution (in unit of m^-2) for a multi‑station interferometer.

    The distribution combines intra‑array baselines from each station and cross‑array baselines
    between stations. For a single station, only the intra‑array contribution is used.

    Parameters
    ----------
    d : array_like
        Baseline lengths (in meters) at which to evaluate the density.
    N : int
        Number of antennas per station.
    D_min : float
        Minimum baseline length (in meters) for a single circular array.
    D_max : float
        Maximum baseline length (in meters) for a single circular array.

    Returns
    -------
    ndarray
        Baseline density in m⁻², with a small epsilon (1e‑10) added to avoid numerical issues.

    Notes
    -----
    The function uses the following external variables (defined in the enclosing scope):
        - n_stations : int
            Total number of stations.
        - unique_Ls : list of float
            Distinct station‑to‑station distances (in meters).
        - mults : list of int
            Multiplicities of each distance in unique_Ls.
    """
    d = np.asarray(d)
    # Intra‑array part: each station contributes C(N,2) baselines
    intra_radial = n_stations * (N * (N - 1) / 2) * single_circular_pdf(d, D_min, D_max)

    # Cross‑array part: sum over all unique distances with multiplicities
    cross_radial = np.zeros_like(d)

    for L_ij, m in zip(unique_Ls, mults):
        cross_radial += m * (N ** 2) * f_cross_length(d, L_ij, D_min, D_max)

    if n_stations == 1:
        total_radial = intra_radial
    else:
        total_radial = intra_radial + cross_radial

    # Convert to uv‑density (m⁻²) as required by the rest of the pipeline
    with np.errstate(divide='ignore', invalid='ignore'):
        uv_density = total_radial / (2 * np.pi * d)
    uv_density[d == 0] = 0
    epsilon = 1e-10
    return uv_density + epsilon

def k_perp_max_func():
    """
    Compute the maximum transverse wavenumber accessible to the interferometer.

    The maximum baseline length is determined as:
        - For a single station: the array diameter.
        - For multiple stations: Diameter of a single array + the longest vertice-to-vertice distance

    Returns
    -------
    float
        Maximum transverse wavenumber in units of Mpc^-1.
    """
    if n_stations == 1:
        max_baseline = D_max
    else:
        max_L = max(unique_Ls) if unique_Ls else 0.0
        max_baseline = max(D_max, max_L + D_max)
    return 2 * np.pi * max_baseline / (r_z * wavelength)


# ======================= Power Spectrum Error ============================
def noise_power_spectrum(k_perp):
    """
    Thermal noise power spectrum as defined in Eq.3.9

    Parameters
    ----------
    k_perp : float
        Perpendicular wavenumber in Mpc^-1

    Returns
    -------
    float
        Thermal noise power spectrum value at this k_perp in units of mK^2 Mpc^3
    """
    u_val = k_perp * r_z / (2 * np.pi)
    D_val = u_val * wavelength

    T_sys = 5000 * (freq / freq_ref) ** (-2.5) #K
    A_eff = wavelength ** 2 * gain / (4 * np.pi) # m^2

    noise_coeff_1 = (wavelength / 1000) * (1 + redshift) * r_z ** 2 / H_z
        ##unit: km Mpc^2 / (km/s/Mpc) = s Mpc^3
    noise_coeff_2 = (wavelength ** 2 / A_eff) ** 2 ##dimensionless
    noise_coeff_3 = (1 / (N_pol * t_tot)) * (S_area / FoV) ##s^-1
    noise_coeff = T_sys ** 2 * noise_coeff_1 * noise_coeff_2 * noise_coeff_3

    nb_D = nb_D_func(D_val, N_antenna, D_min, D_max)
    nb_u = nb_D * wavelength ** 2

    P_noise = noise_coeff / nb_u ##K^2 Mpc^3

    return 1e6 * P_noise ## mK^2 Mpc^3


# ========================== Set up k-bins ==============================
k_perp_min = 2 * np.pi * D_min / (r_z * wavelength)
k_perp_max = k_perp_max_func()

def k_para_max():
    """
    Compute the maximum parallel wavenumber accessible to the survey.

    Returns
    -------
    float
        Maximum parallel wavenumber in units of Mpc^-1.
    """
    freq_21 = (3e8 / 0.211) * 1e-6 ## MHz
    return  (freq_21/(channel_width * 1e-3)) * 2 * np.pi * H_z / (3e5 * (1+redshift) ** 2)

def k_para_min():
    """
    Compute the minimum parallel wavenumber accessible to the survey.

    Returns
    -------
    float
        Minimum parallel wavenumber in units of Mpc^-1.
    """
    bandwidth = freq * bandwidth_frac
    survey_depth_low = results.comoving_radial_distance(freq_to_z(freq + 0.5 * bandwidth))
    survey_depth_high = results.comoving_radial_distance(freq_to_z(freq - 0.5 * bandwidth))

    delta_r = survey_depth_high - survey_depth_low
    return 2 * np.pi / delta_r

k_parallel_max = k_para_max()
k_parallel_min = k_para_min()

k_para_all_edges = np.arange(k_parallel_min, k_parallel_max, dk_para)
k_para_vals = k_para_all_edges[:-1] + 0.5 * dk_para

k_perp_all_edges = 10 ** np.arange(np.log10(k_perp_min), np.log10(k_perp_max), dlnk_perp)
dk_perp = np.array([k_perp_all_edges[kix+1] - k for kix, k in enumerate(k_perp_all_edges[:-1])])
k_perp_vals = k_perp_all_edges[:-1] + 0.5 * dk_perp

k_perp_grid, k_para_grid = np.meshgrid(k_perp_vals, k_para_vals[::-1])

## convert also to k-mu grid for refernce
k_grid = np.sqrt(k_perp_grid ** 2 + k_para_grid ** 2)
mu_grid = np.sqrt(1 - (k_perp_grid / k_grid) ** 2)


# =============== compute the survey volume and number of k-modes in each bin ==========================
def survey_volume():
    """
    Comoving survey volume.

    Returns
    -------
    float
        Comoving survey volume in Mpc^3.
    """
    s_steradian = 0.000304617419786594 * S_area
    s_area_at_z = s_steradian * r_z ** 2 ## Mpc^2

    bandwidth = bandwidth_frac * freq
    delta_z = freq_to_z(freq - 0.5 * bandwidth) - freq_to_z(freq + 0.5 * bandwidth)

    survey_depth = np.absolute(results.comoving_radial_distance(redshift + delta_z) - r_z) ## Mpc
    
    v_survey = s_area_at_z * survey_depth ##Mpc^3

    return v_survey

def find_k_bin_width(kperp, kperp_edges):
    """
    Find the width of the bin it belongs to for a given k_perp value.

    Parameters
    ----------
    kperp : float
        The value of k_perp to locate.
    kperp_edges : 1D numpy array
        A list of values of all bin edges.

    Returns
    -------
    width : float or None
        Width of the bin this k_perp belongs to.
        Returns None if kperp is outside the range.
    """
    i = np.searchsorted(kperp_edges, kperp, side='right')
    idx = i - 1

    width = kperp_edges[idx+1] - kperp_edges[idx]
    return width

def number_k_modes(k_perp, k_para):
    """
    Number of independent Fourier modes in a given k-bin.

    Parameters
    ----------
    k_perp : float
        Perpendicular wavenumber in Mpc^-1.
    k_para : float
        Parallel wavenumber in Mpc^-1.

    Returns
    -------
    float
        Number of modes.
    """
    V_survey = survey_volume() ## Mpc^3

    dk_perp_val = find_k_bin_width(k_perp, k_perp_all_edges)
    dk_para_val = find_k_bin_width(k_para, k_para_all_edges)
    d3k = 2 * np.pi * k_perp * dk_perp_val * dk_para_val

    Nk = d3k * V_survey / (2 * np.pi) ** 3

    return Nk

def delta_Pk(k_perp, k_para):
    '''
    Total power spectrum error in a given k-bin

    Parameters
    ----------
    k_perp : float
        Perpendicular wavenumber in Mpc^-1.
    k_para : float
        Parallel wavenumber in Mpc^-1.

    Returns
    -------
    float
        Total power spectrum error in this k-bin.
    '''
    Nk = number_k_modes(k_perp, k_para)
    P_N = noise_power_spectrum(k_perp)
    Pk_21 = HI_power_spectrum_2D(k_perp, k_para, interp_lis_fid)

    delta_Pk = (1/np.sqrt(Nk)) * (P_N + Pk_21)

    return delta_Pk ## mK^2 Mpc^3

## Compute the 2D power spectrum and its error
deltaPK = delta_Pk(k_perp_grid, k_para_grid)
PS_HI_2D_fid = HI_power_spectrum_2D(k_perp_grid, k_para_grid, interp_lis_fid)


# ================ Fisher matrix forecast =====================================

##set the percentage for numerical derivative
##set a variation for alpha_s
percent = 0.05
alpha_s_vary = 0.05

def get_ps_vary_interp(params):
    """
    Return the interpolators of the 21-cm power spectrum monopole, dipole and quadrupole moments 
        based on a varied cosmology.

    Parameters
    ----------
    params : str
        Name of the parameter to vary (ombh2, omch2, As, ns, H0, tau, alpha_s).

    Returns
    -------
    list of CubicSpline
        Interpolators of monopole, dipole and quadrupole moments for this varied cosmology.
    """
    fid_val = fiducial[params]
    if params == 'alpha_s':
        varied_val = alpha_s_vary
    else:
        varied_val = fid_val * (1 + percent)

    fisher = fiducial.copy()
    fisher[params] = varied_val

    interp_lis_vary = get_HI_power_spectrum_interp(fisher, redshift)

    return interp_lis_vary

def get_basic_cosmology_vary(params):
    """
    Compute true angular diameter distance and Hubble parameter for a varied cosmology
        to be used in Alcock-Pacyznski effect calculations

    Parameters
    ----------
    params : str
        Name of the parameter to vary (ombh2, omch2, As, ns, H0, tau, alpha_s).

    Returns
    -------
    tuple (D_A_true, H_true)
        Angular diameter distance (Mpc) and Hubble parameter (km/s/Mpc).
    """
    fid_val = fiducial[params]
    if params == 'alpha_s':
        varied_val = alpha_s_vary
    else:
        varied_val = fid_val * (1 + percent)

    fisher = fiducial.copy()
    fisher[params] = varied_val

    _, r_z_vary, H_z_vary = get_basic_cosmology(fisher, redshift)
    D_A_true_vary = r_z_vary / (1 + redshift)
    return (D_A_true_vary, H_z_vary)

## put the P(k, p_vary) interpolator and their corresponding true H(z), DA(z) into a dictionary
## each entry is a list of the three moments Pk0, Pk2, Pk4
Pk_interp_vary_dict = {}
cosmo_vary_dict = {}

for pix, param_name in enumerate(fiducial):
    interp_vary = get_ps_vary_interp(param_name)
    Pk_interp_vary_dict[param_name] = interp_vary
    
    cosmo_vary_tuple = get_basic_cosmology_vary(param_name)
    cosmo_vary_dict[param_name] = cosmo_vary_tuple


def fisher_element(param1, param2):
    """
    Compute a single entry in the Fisher matrix.

    Parameters
    ----------
    param1, param2 : str
        Names of the two cosmological parameters (ombh2, omch2, As, ns, H0, tau, alpha_s).

    Returns
    -------
    float
        Fisher matrix entry corresponding to these two parameters.
    """
    # Get the true distances for the varied cosmologies
    D_A_true1, Hz_true1 = cosmo_vary_dict[param1]
    D_A_true2, Hz_true2 = cosmo_vary_dict[param2]

    # Compute ratios relative to the fixed fiducial values
    D_A_ratio1 = D_A_fid / D_A_true1
    Hz_ratio1 = Hz_true1 / Hz_fid

    D_A_ratio2 = D_A_fid / D_A_true2
    Hz_ratio2 = Hz_true2 / Hz_fid

    ## param1
    interp_lis_vary_1 = Pk_interp_vary_dict[param1]
    PS_vary_1 = HI_power_spectrum_2D(k_perp_grid, k_para_grid, interp_lis_vary_1, D_A_ratio1, Hz_ratio1)

    ## param2
    interp_lis_vary_2 = Pk_interp_vary_dict[param2]
    PS_vary_2 = HI_power_spectrum_2D(k_perp_grid, k_para_grid, interp_lis_vary_2, D_A_ratio2, Hz_ratio2)

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


## build the Fisher matrix
fiducial_fisher = fiducial.copy()
fiducial_fisher.pop('tau')

fisher_matrix = np.zeros((len(fiducial_fisher), len(fiducial_fisher)))

def run():
    """
    Main driver: 
        - compute Fisher matrix
        - save outputs
        - print constraints
        - print number of modes.
    """
    for pix1, p1 in enumerate(fiducial_fisher):
        for pix2, p2 in enumerate(fiducial_fisher):
            entry = fisher_element(p1, p2)
            fisher_matrix[pix1, pix2] = entry

    ## same the Fisher matrix, power spectrum and error
    np.savez(f'outputs_{str(n_stations)}_array.npz',
             fisher=fisher_matrix,
             ps_21=PS_HI_2D_fid,
             ps_21_error=deltaPK,
             kperp=k_perp_vals,
             kpara=k_para_vals)
    logging.info(f"Saved outputs to outputs_{str(n_stations)}_array.npz")
    
    ## Compute the covariance matrix and parameter constraint
    cov = np.linalg.inv(fisher_matrix)

    logging.info('------------------------------------')
    logging.info('Constraint on cosmological parameters:')

    for pix, params in enumerate(fiducial_fisher):
        err = np.sqrt(cov[pix, pix])
        logging.info(f'{params}: {err}')

    ## calculate the number of independent modes 
    N_modes = np.sum((PS_HI_2D_fid/deltaPK)**2)

    logging.info('------------------------------------')
    logging.info(f'Number of modes: {N_modes}')   

if __name__ == "__main__":
    run()
