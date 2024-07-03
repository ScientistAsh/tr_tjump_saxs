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
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
from numpy.linalg import svd
import sys

import seaborn as sns
import matplotlib.pylab as pl 
from time import sleep
from tqdm.notebook import tqdm
import csv
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d

from file_handling import *


import numpy as np
from scipy.optimize import curve_fit


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

# Function to load pdb data
def load_pdb(file):
    '''
    This function will load a single PDB file.  
    
    Parameters:
    -----------
    file : str
        String for the full path to the PDB file to be loaded.  
        
    Returns:
    --------
     p : MDAnalysis.Universe
        An MDAnalysis Universe object containing the loaded PDB structure.

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
    pdb = load_pdb(file='path/to/your/file.pdb')
    '''
    # set formating parameters
    red = "\033[131m"
    green = "\033[1;32m"
    reset = "\033[0m"

    # load pdb file
    try:
        if not os.path.exists(file):
            raise FileNotFoundError(red + f"The file {file} does not exist." + reset)
        
        p = mda.Universe(file)
        print(green + "PDB file loaded successfully!" + reset)
        return p

    # exception raising
    except FileNotFoundError as fnf_error:
        print(red + fnf_error + reset)
    except IOError as io_error:
        print(red + f"Error reading the file {file}: {io_error}" + reset)
    except ValueError as val_error:
        print(red + f"Error parsing the file {file}: {val_error}" + reset)
    except Exception as e:
        print(red + f"An unexpected error occurred: {e}" + reset)

# function to load a set of PDBs
def load_pdb_set(directory):
    '''
    This function will load a set of PDB files.  
    
    Parameters:
    -----------
    directory : str
        String for the full path to the directory storing the PDB files.  
        
    Returns:
    --------
     structures : list
        A list in which each entry is an MDAnalysis Universe object containing the loaded PDB structure.

    Raises:
    -------
    FileNotFoundError
        If the directory does not exist.
    IOError
        If there is an error reading the files in the directory.
        
    Examples:
    ---------
    pdb = load_pdb_set(directory='path/to/your/directory/')
    '''
    
    # set formating parameters
    red = "\033[131m"
    green = "\033[1;32m"
    reset = "\033[0m"
    
    structures = []

    try:
        if not os.path.exists(directory):
            raise FileNotFoundError(red + f"The directory {directory} does not exist." + reset)
        
        for filename in os.listdir(directory):
            if filename.endswith(".pdb"):
                file_path = os.path.join(directory, filename)
                try:
                    u = mda.Universe(file_path)
                    structures.append(u)
                    print('PDB file loaded successfully!')
                    
                except Exception as e:
                    print(red + f"Error loading file {file_path}: {e}" + reset)

        print(green + 'Successfully loaded ' + str(len(structures)) + ' PDB files.' + reset)

    except FileNotFoundError as fnf_error:
        print(red + fnf_error + reset)
    except IOError as io_error:
        print(red + f"Error reading files in directory {directory}: {io_error}" + reset)
    except Exception as e:
        print(red + f"An unexpected error occurred: {e}" + reset)
        

          
    return structures