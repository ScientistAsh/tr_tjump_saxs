'''
tr_tjump_saxs: saxs_qc.py
Date created: 31 July 2023

This module is part of the tr_tjump_saxs package for processing and anlayzing pump-probe time resolved, 
temperature-jump small angle X-ray scattering data sets. This module includes functions for kinetic
analysis on TR, T-Jump SAXS data. 

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

def saxs_auc(flist, times=[1.5, 3, 5, 10, 50, 100, 300, 500, 1000],
             delim=',', qmin=None, qmax=None, outdir=None,
             data_outfile=None, plot_outfile=None,
             xlab='Time Delay ($\mu$s)',
             plot_title='CH505TF SAXS T-Jumps Area Under the Curve'):
    
    '''
    Function to calculate the area under the curve for 
    solution x-ray scattering difference curves. The 
    calculated AUC and it's corresponding file will be 
    saved as a CSV file. If you would like to continue 
    to use the AUC data for further analysis call
    the function inside a variable definition. Plots of 
    the calculated AUC as a function of times will be saved. 
        
    Parameters:
    -----------
    flist : list
        List of files, including full path,  containing curves 
        to be loaded and averaged. When set to none, will load 
        files with given prefix/suffix from the indicated data_dir. 
        Default is None. Note that either flist or the directory 
        storing files must be given. 
    
    times (optional) : list
        List of times corresponding to the data to run AUC analysis. 
        Will be used as the X-label values for plots. The default value
        is [1.5, 3, 5, 10, 50, 100, 300, 500, 1000].
        
    delim (optional) : str
        Delimitter used in data files. Default is comma. 
    
    qmin : float
        Q value to begin integration at. If set to None, the will start at lowest
        Q value in input data set. Default value is None. 
        
    qmax : float
        Q value to end integration at. If set to None, the will end at highest
        Q value in input data set. Default value is None. 
    
    outdir (optional) : str
        Directory to save calculated area under the curve calculations and plots 
        to. When set to None, no files will be saved. Default is None. 
        
    data_outfile (optional) : str
        Name of saved file storing the calculated average curve. Will be saved in 
        outdir as PNG. When set to None, no file will be saved. Defualt is None.
        
    plot_outfile (optional) : str
        Name of saved file containing the plot. File will be saved as PNG in outdir. 
        When set to None, no file will be saved.
        
    xlab (optional) : str
        Label to use for X-axis in plot. Default value is 'Time Delay (us).'
    
    plot_title (optional) : str
        Label to use for title of plot. Default value is 'CH505TF SAXS T-Jumps Area Under the Curve'
        
        
    Returns:
    --------
    auc : np.array
        Contains the time delay and calculated AUC values. 
        
    Raises:
    -------
    ImportError :
        When the scipy package is not found
    FileNotFoundError : 
        When input file is not found
    ValueError : 
        When input file format or delimiter is 
        wrong
        
    Examples:
    ----------
    ex.1: Continue using data (data will also be saved):
    a = saxs_auc(flist=files, delim=',', q_min=0, 
                q_max=0.3, 
                outdir='../../ANALYSIS/NALYSIS/AUC/', 
                data_outfile='tjumps_auc.csv',
                plot_outfile='tjump_auc_plot.png')
                
    ex.2: Just save data to csv file:
    saxs_auc(flist=files, delim=',', q_min=0, q_max=0.3)
    '''
    
    
    try:
        import scipy
    except ImportError:
        raise ImportError("The 'scipy' module is required to run this function.")

    def make_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def load_saxs(file, delim=',', mask=0):
        try:
            data = np.genfromtxt(file, delimiter=delim, skip_header=mask)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{file}' not found.")
        except ValueError:
            raise ValueError(f"Error while reading data from '{file}'. Check the file format and delimiter.")
        return data

    try:
        # rest of the function code goes here...

        # set variable names
        auc = []
        data = []

        # loop over difference files for time_delay in directory
        for t, f in zip(times, flist):
            print('Loading ' + str(f))

            # load data
            data = load_saxs(file=f, delim=delim, mask=0)

            # filter q range for integration
            filtered = data[(data[:,0] >= 0.02) & (data[:,0] <= 0.3)]

            # calculate area under the curve
            area = scipy.integrate.simpson(filtered[:,1], filtered[:,0])

            # append area and time delay to auc list
            auc.append([t, area])

        # convert auc list to array
        auc = np.asarray(auc)

        # store auc calculations to file
        if outdir is not None:
            make_dir(outdir)
            np.savetxt(str(outdir) + '/' + str(data_outfile), 
                   auc, delimiter=",")

        # plot AUC
        ax = plt.axes([0.125,0.125, 5, 5])
        plt.plot(times, auc[:, 1], linewidth=10, alpha=0.3)
        plt.scatter(times, auc[:, 1], marker='o', s=500.)
        plt.set_cmap('rainbow')
        plt.xlabel(xlab, fontsize=50, fontweight='bold')
        plt.ylabel('Area Under the Curve (Simpsons)', fontsize=50, fontweight='bold')
        plt.xticks(fontsize=45)
        plt.yticks(fontsize=45)
        plt.title(plot_title, fontsize=70, fontweight='bold')

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(5)

        # save figure
        if outdir is not None and plot_outfile is not None:
            plt.savefig(str(outdir + plot_outfile), bbox_inches='tight')

        plt.show()

    except Exception as e:
        # Catch any exception that might occur during the execution of the function
        print("An error occurred:", str(e))

    return auc