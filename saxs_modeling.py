'''
tr_tjump_saxs: saxs_qc.py
Date created: 7 August 2023

This module is part of the tr_tjump_saxs package for processing and anlayzing pump-probe time resolved, 
temperature-jump small angle X-ray scattering data sets. This module includes functions for modeling
TR, T-Jump and static SAXS data. 

Ashley L. Bennett, PhD
@ScientistAsh
'''

# import dependent modules
import numpy as np
import scipy
import os 
import pandas as pd
from pandas import read_table,DataFrame
from collections import namedtuple
import shutil
import math
import warnings
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
from numpy.linalg import svd
import sys
import seaborn as sns
import matplotlib.pylab as pl 
from time import sleep
from tqdm.notebook import tqdm
import csv
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from file_handling import *
from scipy.optimize import curve_fit
#import MDAnalysis as mda

def delta_pr(curve1, curve2, delim1=None, delim2=None, skip1=None, 
            skip2=None, kind='linear', fill_value='extrapolate',
            outdir=None, outfile=None):
    '''
    This function calculates the difference between two distance
    distribution functions (P(r)). Function assumes the x-values 
    of the two input curves are not identical and automatically
    interpolates the x-values. Difference is calculated as:
                            curve1 - curve2
    
    Function will save a CSV file of the delta P(r) curve and a PNG file
    of the delta P(r) curve plot if outfile is specified. 
    
    Parameters
    -----------
    curve1 : np.array
        Array containing PDDF data. 
        
    curve 2 : np.array
        Array containing PDDF data. 
 
    delim1 (optional) : str
        Delimitter used in curve1 file. Default value is None.
        Ex: ',' for comma delimitted, ' ' for space delimitted, and '\t' for tab
        delimitted.
    
    delim2 (optional) : str
        Delimitter used in curve2 file. Default value is None.
        Ex: ',' for comma delimitted, ' ' for space delimitted, and '\t' for tab
        delimitted.
        
    skip1 (optional) : int
        Number of rows to skip when importing curve1 data. Default value is None.
        
    skip2 (optional) : int
        Number of rows to skip when importing curve1 data. Default value is None.
        
    kind (optional) : str
        Specifies the kind of interpolation as a string or as an integer specifying 
        the order of the spline interpolator to use. The string has to be one of 
        ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, 
        ‘previous’, or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a 
        spline interpolation of zeroth, first, second or third order; 
        ‘previous’ and ‘next’ simply return the previous or next value of the point; 
        ‘nearest-up’ and ‘nearest’ differ when interpolating half-integers (e.g. 0.5, 1.5) 
        in that ‘nearest-up’ rounds up and ‘nearest’ rounds down. Default is ‘linear’.
        (From scipy.interpolation.interp1d docstrings: 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html). 
        Kind for both curve1 and curve2 are the same. 
    
    fill_value (optional) : array-like or (array-like, array_like) or “extrapolate”
        if a ndarray (or float), this value will be used to fill in for requested 
        points outside of the data range. If not provided, then the default is NaN. 
        The array-like must broadcast properly to the dimensions of the non-interpolation 
        axes. 
        If a two-element tuple, then the first element is used as a fill value for 
        x_new < x[0] and the second element is used for x_new > x[-1]. Anything that is 
        not a 2-element tuple (e.g., list or ndarray, regardless of shape) is taken to be 
        a single array-like argument meant to be used for both bounds as below, 
        above = fill_value, fill_value. Using a two-element tuple or ndarray requires 
        bounds_error=False. 
        (From scipy.interpolation.interp1d docstrings: 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html).
        fill_value for both curve1 and curve2 are the same. 
        
    outdir (optional) : str
        Full path to directory to store output files in. If the directory does not already exist
        it will be made. When set tyo None, then no files will be saved. The default value is None. 
        
    outfile (optional) : str
        File name, including full path, to store output files. Saved output files include the 
        delta P(r) curve contained in a CSV file and a png plot. 
        
    Examples:
    ---------
    dpr1 = delta_pr(curve1=noApex, curve2=closed, delim1=',', delim2=',', skip1=2, skip2=2, 
               kind='linear', fill_value='extrapolate', 
               outfile='../../ANALYSIS/THEORETICAL/DELTA_PR/3closed_NoApex-3closed_man9.csv')

    '''
    # load P(r) curves
    #curve1 = np.loadtxt(fname=curve1, delimiter=delim1, skiprows=skip1)
    #curve2 = np.loadtxt(fname=curve2, delimiter=delim2, skiprows=skip2)
    
    # Create interpolation functions for both curves
    f1 = interp1d(curve1[:, 0], curve1[:, 1], kind=kind, fill_value=fill_value)
    f2 = interp1d(curve2[:, 0], curve2[:, 1], kind=kind, fill_value=fill_value)
    
    # Merge the x values of both curves
    x = np.unique(np.concatenate((curve1[:, 0], curve2[:, 0])))
    
    # Compute the difference between y values of both curves
    diff = f1(x) - f2(x)
    
    # Combine the x values and the corresponding differences into the resulting curve
    delta_pr = np.array([x, diff])   
    
    # plot delta p(r)
    ax = plt.axes([0.125,0.125, 5, 5])
    ax.tick_params(which='major', length=20, width=5, direction='out')
        
    plt.plot(delta_pr[0], delta_pr[1], linewidth=10)
    plt.xlabel('$\Delta$ Distance (Å)', fontsize=60, fontweight='bold')
    plt.ylabel('$\Delta$P(r)', fontsize=60, fontweight='bold')
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.title('Distance Difference Distribution', fontsize=70, fontweight='bold')  
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(5)
        
    # save files
    if outdir is not None:
        make_dir(f=outdir)
        np.savetxt(fname=str(outdir + outfile) + '.csv', X=np.c_[delta_pr[0], delta_pr[1]], delimiter=',')    
        plt.savefig(str(outdir + outfile) + '.png', bbox_inches='tight')

    plt.show()
        
    
    return delta_pr

def load_structure(file, topology_file=None):
    '''
    This function will load a structure file in either PDB or trajectory format.
    
    Parameters:
    -----------
    file : str
        String for the full path to the structure file to be loaded (PDB or trajectory).
    topology_file : str, optional
        Path to the topology file (required for trajectory files like DCD).
        
    Returns:
    --------
    u : MDAnalysis.Universe
        An MDAnalysis Universe object containing the loaded structure.
        
    Raises:
    -------
    FileNotFoundError
        If the file does not exist.
        
    IOError
        If the file cannot be read.
        
    ValueError
        If the file format is incorrect or the file cannot be parsed.
        
    Examples:
    ---------
    u = load_structure(file='path/to/your/file.pdb')
    u = load_structure(file='path/to/your/trajectory.dcd', topology_file='path/to/topology.psf')
    '''
    # Set formatting parameters
    red = "\033[131m"
    green = "\033[1;32m"
    reset = "\033[0m"

    # Load structure file
    try:
        if not os.path.exists(file):
            raise FileNotFoundError(red + f"The file {file} does not exist." + reset)
        
        # Check if it's a PDB file
        if file.endswith('.pdb'):
            u = mda.Universe(file)
            print(green + "PDB file loaded successfully!" + reset)
        
        # Check if it's a trajectory file
        elif file.endswith(('.dcd', '.xtc', '.trr', '.nc')):
            if topology_file is None:
                raise ValueError(red + "Topology file is required for trajectory files." + reset)
            if not os.path.exists(topology_file):
                raise FileNotFoundError(red + f"The topology file {topology_file} does not exist." + reset)
            
            u = mda.Universe(topology_file, file)
            print(green + "Trajectory file loaded successfully!" + reset)
        
        else:
            raise ValueError(red + f"Unsupported file format: {file}" + reset)
        
        return u

    # Exception handling
    except FileNotFoundError as fnf_error:
        print(red + str(fnf_error) + reset)
    except IOError as io_error:
        print(red + f"Error reading the file {file}: {io_error}" + reset)
    except ValueError as val_error:
        print(red + f"Error parsing the file {file}: {val_error}" + reset)
    except Exception as e:
        print(red + f"An unexpected error occurred: {e}" + reset)


# function to load a set of PDBs
def load_ensemble(files, topology_files=None):
    '''
    This function will load a single or a list of PDB or trajectory files.
    
    Parameters:
    -----------
    files : str or list of str
        A single file path or a list of file paths to the structure files to be loaded.
    topology_files : str or list of str, optional
        Path(s) to the topology file(s) corresponding to the trajectory files, if applicable.
        If a single topology file is provided, it will be used for all trajectory files.
        
    Returns:
    --------
    structures : list
        A list in which each entry is an MDAnalysis Universe object containing the loaded structure.
        
    Raises:
    -------
    FileNotFoundError
        If any of the files do not exist.
    ValueError
        If a topology file is required but not provided.
        
    Examples:
    ---------
    # Load a single PDB file
    structures = load_structures(files='path/to/your/file.pdb')
    
    # Load multiple PDB or trajectory files
    structures = load_structures(files=['path/to/your/file1.pdb', 'path/to/your/file2.dcd'], 
                                 topology_files='path/to/your/topology.psf')
    '''
    
    # Set formatting parameters
    red = "\033[131m"
    green = "\033[1;32m"
    reset = "\033[0m"
    
    # Ensure `files` is a list, even if a single file is provided
    if isinstance(files, str):
        files = [files]
    
    # Ensure `topology_files` is a list if multiple trajectory files are provided
    if isinstance(topology_files, str):
        topology_files = [topology_files] * len(files)
    
    structures = []

    try:
        for i, file_path in enumerate(files):
            if not os.path.exists(file_path):
                raise FileNotFoundError(red + f"The file {file_path} does not exist." + reset)
            
            # Load PDB files
            if file_path.endswith(".pdb"):
                try:
                    u = mda.Universe(file_path)
                    structures.append(u)
                    print(green + f'PDB file {file_path} loaded successfully!' + reset)
                    
                except Exception as e:
                    print(red + f"Error loading PDB file {file_path}: {e}" + reset)
            
            # Load trajectory files
            elif file_path.endswith(('.dcd', '.xtc', '.trr', '.nc')):
                if topology_files is None:
                    raise ValueError(red + f"Topology file is required for trajectory file {file_path}." + reset)
                
                topology_file = topology_files[i]
                if not os.path.exists(topology_file):
                    raise FileNotFoundError(red + f"The topology file {topology_file} does not exist." + reset)
                
                try:
                    u = mda.Universe(topology_file, file_path)
                    structures.append(u)
                    print(green + f'Trajectory file {file_path} loaded successfully with topology {topology_file}!' + reset)
                    
                except Exception as e:
                    print(red + f"Error loading trajectory file {file_path}: {e}" + reset)
            
            else:
                print(red + f"Unsupported file format: {file_path}. Skipping file." + reset)
        
        print(green + f'Successfully loaded {len(structures)} structure files.' + reset)

    except FileNotFoundError as fnf_error:
        print(red + str(fnf_error) + reset)
    except ValueError as val_error:
        print(red + str(val_error) + reset)
    except Exception as e:
        print(red + f"An unexpected error occurred: {e}" + reset)
    
    return structures


def initialize_weights_uniform(n_structures):
    '''
    Initialize weights to be uniformly distributed to fit theoretical SAXS difference curves to experimental SAXS
    difference curves.

    Parameters:
    ------------
    n_structures : int 
        The number of structures in the ensemble.
        
    Returns:
    ---------
    Array of weights initialized to uniform values.
    '''
    return np.ones(n_structures) / n_structures



def bayesian_fit(weights, difference_curves, experimental_saxs):
    """
    Objective function to minimize the chi-squared difference and maximize entropy. 
    Chi-squared is calculated according to:
             chi_squared = np.sum((calculated_saxs - experimental_saxs) ** 2)
             
    The maximum entropy principle is defined by:
                entropy = -np.sum(weights * np.log(weights + 1e-10))

    Parameters:
    -----------
    weights : np.array() 
        Array of weights for the structures.
        
    difference_curves : np.array
        Precomputed difference curves for the ensemble.
        
    experimental_saxs : np. array()
        Experimental SAXS difference curve.
        
    Returns:
    ---------
    Value of the objective function.
    """ 
    
    calculated_saxs = np.sum(weights[:, None] * difference_curves, axis=0)
    
    chi_squared = np.sum((calculated_saxs - experimental_saxs) ** 2)
    
    # Entropy term for maximum entropy principle
    entropy = -np.sum(weights * np.log(weights + 1e-10))  # Small value to avoid log(0)
    
    # Objective: minimize chi_squared, maximize entropy (negative sign for entropy)
    return chi_squared - entropy

