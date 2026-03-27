# DarkAgesCosmology
Fisher forecast of cosmological parameters based on 21-cm power spectrum measurement during the cosmic Dark Ages 

This repository provides the code associated with the paper:

**Title: A Designer’s Guide to Lunar Far-Side Interferometer Array: Power Spectrum Measurement and Cosmological Constraints from the Dark Ages**  
Authors: Yuewei Wen and Xuelei Chen

## Features
- **21‑cm power spectrum**: Compute the 2D 21cm power spectrum of the during the Dark Ages (30 < z < 200).
- **Interferometer noise**: Model the thermal noise of a lunar far-side interferometer array with the choice between two different configurations:
  - A single circular array
  - Two identical circular arrays separated by a distance and connected through interferometry
- **Forecast of Cosmological Constraints**: Use the Fisher information matrix formalism to predict constraints on cosmological parameters plus the running of the spectral index $\alpha_s$ based on user's choice of array set-up.

## Structure
- `dark_ages_21cm_single_array.py` – core calculation for the single-array configuration
- `dark_ages_21cm_double_array.py` – core calculation for the double-array configuration
- `params.ini` – parameter file
- `README.md` – documentation

## Requirements
- Python 3.x
- numpy
- scipy
- matplotlib
- camb

## Quick Start

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/DarkAgesCosmology.git
cd DarkAges

python matter_power_21cm_fisher_z_single.py

or

python matter_power_21cm_fisher_z_double.py

## Citation
@article{YourLastName2025,
  title   = {A Designer's Guide to Lunar Far‑Side Interferometer Arrays: Power Spectrum Measurement and Cosmological Constraints from the Dark Ages},
  author  = {Yuewei Wen, Xuelei Chen},
  journal = {...},
  year    = {2026},
  note    = {to appear}
}
