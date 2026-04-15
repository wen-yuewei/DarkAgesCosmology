# DarkAgesCosmology

Fisher forecast of cosmological parameters based on 21‑cm power spectrum measurements during the cosmic Dark Ages.

This repository provides the code associated with the paper:

**Title:** *A Designer’s Guide to Lunar Far‑Side Interferometer Array: Power Spectrum Measurement and Cosmological Constraints from the Dark Ages*  
**Authors:** Yuewei Wen et al. (2026)

---

## Features

- **21‑cm power spectrum** – Computes the 2D power spectrum of the 21‑cm signal during the Dark Ages (z ≈ 30–200).
- **Interferometer noise** – Fully general thermal noise modeling: any number of identical circular stations (from one to many) can be placed at vertices of a regular n-gon. 
- **Cosmological forecasts** – Uses the Fisher matrix formalism to predict constraints on cosmological parameters (Ω<sub>b</sub>h², Ω<sub>c</sub>h², A<sub>s</sub>, n<sub>s</sub>, H<sub>0</sub>) and the running of the spectral index α<sub>s</sub>.

---

## Structure
- `dark_ages_21cm_master.py` – core calculations
- `params.ini` – parameter file
- `README.md` – documentation

---

## Requirements

- Python 3.7+
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [CAMB](https://camb.info/) (tested with version 1.5+)

---

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/wen-yuewei/DarkAgesCosmology.git
   cd DarkAgesCosmology
2. Edit `params.ini` to set your desired parameters.
3. Run the master script `dark_ages_21cm_master.py`.

---

## Configuration (`params.ini`)

The parameter file is divided into sections. The most important choices are:

- **`n_stations`** – choose the number of stations/sub-arrays (from 1 to n).
- **`D_min`**, **`D_max`** – minimum and maximum baseline length limits (in meters); when `n_stations` > 1, **`D_max`** refers to the diameter of each station/sub-array. 
- **`N_antenna`** – number of antennas; for `n_stations` > 1, this is the number per array.
- **`L`** – side length of the n-gon formed by stations/sub-arrays (only used when `n_stations` > 1).
- **Cosmological parameters** – Planck 2018 best‑fit values are given as defaults.
- **Survey parameters** – `t_tot`, `S_area`, etc., and k‑binning settings.

All units are explained in the comments inside the file.

---

## Usage
Simply run the master script from the terminal:

```bash
python dark_ages_21cm_master.py
```

The script will:

- Load the parameters from params.ini.
- Compute the 21‑cm power spectrum and its expected error.
- Build the Fisher matrix.
- Print the 1‑σ marginalized constraints on the cosmological parameters and the total number of modes probed by this survey to the terminal.
- Save the Fisher matrix, power spectrum, its error and k‑bin centers to a `.npz` file named `outputs_{n_stations}_array.npz`.

Example output:

```
Saved outputs to outputs_3_array.npz
-----------------------
Constraint on cosmological parameters:
ombh2: 0.004256608688653717
omch2: 0.016240351726979102
As: 1.6487225295945772e-09
ns: 0.08244841267294853
H0: 3.3892816894726923
alpha_s: 0.05565676991265285
-----------------------
Number of modes: 81577.67147385971
```

---

## Output Files

The script saves a compressed NumPy archive (`.npz`) with the following:

- **`fisher`** – Fisher matrix (size n×n, where n is the number of varied parameters).
- **`ps_21`** – 2D power spectrum (grid of shape `len(k_para_vals) × len(k_perp_vals)`).
- **`ps_21_error`** – Error on the power spectrum.
- **`kperp`** – 1D array of k<sub>⊥</sub> bin edges (in Mpc⁻¹).
- **`kpara`** – 1D array of k<sub>∥</sub> bin edges (in Mpc⁻¹).

To load the data in another Python session:

```python
import numpy as np
data = np.load('outputs_single_array.npz')
fisher = data['fisher']
ps = data['ps_21']
# etc.
```

---

## Citation
If you use this code for your research, please cite the paper:

```Bibtex
@article{Wen2026,
  title   = {A Designer's Guide to Lunar Far‑Side Interferometer Array: Power Spectrum Measurement and Cosmological Constraints from the Dark Ages},
  author  = {Yuewei Wen et al.},
  journal = {to appear},
  year    = {2026}
}
```
