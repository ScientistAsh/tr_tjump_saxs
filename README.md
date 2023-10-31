# tr_tjump_saxs
**Language:** Python3 <br>
**Lincense:** BSD-3 Clause <br>
**Developer:** @ScientistAsh <br>
**Updated:** 20 October 2023 <br>
**v1.0.0** [![DOI](https://zenodo.org/badge/622369035.svg)](https://zenodo.org/doi/10.5281/zenodo.10028766) <br>

## Description
Python package to process and analyze pump-probe time resolved, temperature-jump small angle X-ray scattering data.

## Dependecies
- numpy=1.20.1=py38h93e21f0_0
- numpy-base=1.20.1=py38h7d8b39e_0
- numpydoc=1.1.0=pyhd3eb1b0_1
- scipy=1.7
- pandas=1.2.4=py38h2531618_0
- sortedcollections=2.1.0=pyhd3eb1b0_0
- backports.shutil_get_terminal_size=1.0.0=pyhd3eb1b0_3
- mpmath=1.2.1=py38h06a4308_0
- matplotlib=3.3.4=py38h06a4308_0
- matplotlib-base=3.3.4=py38h62a2d02_0
- seaborn=0.11.1=pyhd3eb1b0_0
- tqdm=4.59.0=pyhd3eb1b0_1
- unicodecsv=0.14.1=py38_0
- datetime

## Environment 
The environment used to develop this package can be cloned from the environment.yml file. To clone the environment:

`conda env create -f environment.yml` 

The name of the environment is tr_tjump_saxs and is determined by the first line in the environment.yml file. 
If you would like to change the name, replace tr_tjump_saxs with your preferred name in the first line of the environment.yml file. 

## Install
To get this code, simply clone the git repository:

`git clone  https://github.com/ScientistAsh/tr_tjump_saxs.git`

## Usage 
Extensive docstrings are provided in the source code to explain the usage, input parameters, output, and examples.
For more extensive discussion on how to use the code, tutorial files are provided in the TUTORIALS. 
The tutorials are in Jupyter Notebooks and allow for interactive usage (note that you must have SAXS data for interactive usage).
It is recommended to either use Jupyter Notebooks or to write scripts and run them non-interactively. 

This package is broken down into several different modules, each of which focuses on a specific aspect of the SAXS data processing/analysis workflow. 

### `file_handling`
The `file_handling` module contains functions for loading and storing files. The functions in the module include:
- `make_dir()` to make new directories
- `make_flist()` to make a list of files
- `unique_set()` to get the unque values in a list
- `sort_key()` to sort file lists according to time delays
- `load_saxs()` to load a single SAXS scattering or difference curve
- `lost_set()` to load a set of SAXS scattering or difference curves
- `plot_curve` to plot SAXS scattering or difference curves

### `saxs_processing`
The `saxs_processing()` module contains functions to remove outliers, scale, and subtract data. The function in the module include:
- `svd_outliers()` uses an SVD method to detect outliers in SAXS scattering curves (could also be used for difference curves)
- `iterative_chi()` uses a chi-square method to detect outliers and repeats the process until no outliers are detected
- `remove_outliers()` removes a set of previously determined outliers from a file list
- `saxs_scale()` scales a SAXS scattering or difference curve to a reference curve
- `saxs_sub()` subtracts a SAXS scattering or difference curve from a reference curve
- `auc_outlier()` detects outliers using and area under the curve method (likely to be deprecated) 
- `move_outliers()` moves outlier files to a new directory (likely to be deprecated)

### `saxs_qc`
The `saxs_qc` module provides functions to assess the quality of SAXS data. A function to calculate the PDDF will be added (eventually). The functions in this module include:
- `guinier_analysis()` performs a guinier analysis
- `sys_err()` assess the data for systematic errors during the experimental run
- `kratky_plot()` performs a kratky analysis 
- `temp_cal()` uses a linear regression method to estimate the temperature jump

### `saxs_kinetics`
The `saxs_kinetics` module provides functions to determine the kinetics from SAXS T-Jump data. The functions in this module include:
- `saxs_auc()` uses the Simpson's method to determine the area under the curve for a given set of SAXS difference curves
- `svd_kinetics()` runs an SVD analysis on a set of input SAXS difference curves
- `auc_fit()` fits the area under the curve data to exponential equations
- ` svd_fit()` fits the SVD right vectors to exponential equations

### `saxs_modeling`
The `saxs_modeling` module provides modeling saxs data. Functions will be added to this module in the future. The functions currently in this module include:
- `delta_pr` uses an interpolation method to subtract a P(r) curve from a reference P(r) curve

## Data avialbility
*COMING SOON*

## Script availability 
*COMING SOON*

## Citing
For publications and presentations please ackwoledge the use of this package in the ackowledgements. 
Please also cite the associated publication:

*Microsecond dynamics control the HIV-1 envelope conformation
Ashley L. Bennett, R.J. Edwards, Irina Kosheleva, Carrie Saunders, Yishak Bililign, Ashliegh Williams, Katayoun Manosouri, 
Kevin O. Saunders, Barton F. Haynes, Priyamvada Acharya, Rory Henderson
bioRxiv 2023.05.17.541130; doi: https://doi.org/10.1101/2023.05.17.541130*

