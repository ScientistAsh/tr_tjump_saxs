'''
tr_tjump_saxs: saxs_qc.py
Date created: 26 July 2023

This module is part of the tr_tjump_saxs package for processing and anlayzing pump-probe time resolved, 
temperature-jump small angle X-ray scattering data sets. This module includes functions for assessing 
SAXS data quality. Functions were validated using results from ATSAS Primus. 

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

def guinier_analysis(file, label, delim=',', mask=0, qmax=0.0025):
    '''
    Perform Guinier analysis on a SAXS scattering curve.

    Parameters:
    ------------
        file : str
            File containing scattering data to 
            run Guinier analysis on.
            
        label : str
            Name of dataset to be used in plot titles.
        
        delim (optional) : str
            Type of delimitter used in input files. Default 
            value is comma (','). 
            
        mask (optional) : int
            Number of rows to skip when importing numpy array.
            When set to 0 all rows ar imported. Default value
            is 0.    
            
        qmax (optional) : float
            Maximum q value to include in guinier fitting. Default
            value is 0.0025Å-1.

    Returns:
    --------
        Rg : float
            Radius of gyration (Rg) in units of inverse q.
            
        Rg_error_scaled : float
            Error associated with Rg.
            
        I_0 : float
            The calculated I0.
            
        I_0_error_scaled : float
            Error of I0
            
    Rasies:
    -------
    
    
    Examples:
    ---------
    temps = ['30C', '35C', '40C', '44C']
    static_files = ['/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/ANALYSIS/STATIC/BUFFER_SUB/30C_buffsub.csv',
                    '/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/ANALYSIS/STATIC/BUFFER_SUB/35C_buffsub.csv',
                    '/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/ANALYSIS/STATIC/BUFFER_SUB/40C_buffsub.csv',
                    '/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/ANALYSIS/STATIC/BUFFER_SUB/44C_buffsub.csv']
                    
    for t, f in zip(temps, static_files):
        print('Running Guinier Analysis on ' + str(t))
        guinier_analysis(file=str(f), label=str(t), delim=',', mask=0, qmax=0.0025)
        
    > Running Guinier Analysis on 30C
      Loading data...
      Running Guinier Analysis...
      Rg  = 54.12 +/- 0.40
      I_0 = 12.09 +/- 0.24
      Fitting model...
      Plotting Data...

      
      Running Guinier Analysis on 35C
      Loading data...
      Running Guinier Analysis...
      Rg  = 54.11 +/- 0.40
      I_0 = 11.99 +/- 0.23
      Fitting model...
      Plotting Data...
      
      Running Guinier Analysis on 40C
      Loading data...
      Running Guinier Analysis...
      Rg  = 54.12 +/- 0.39
      I_0 = 11.95 +/- 0.23
      Fitting model...
      Plotting Data...
      
      Running Guinier Analysis on 44C
      Loading data...
      Running Guinier Analysis...
      Rg  = 53.84 +/- 0.32
      I_0 = 11.87 +/- 0.19
      Fitting model...
      Plotting Data...
    '''
    
    # load data 
    print('Loading data...')
    curve = load_saxs(file=file, delim=delim, mask=mask)
    
    # define guinier equation
    def linear(x,a,b):
        return b-a*x
    
    # select qmin and qmax
    print('Running Guinier Analysis...')
    qmin = np.min(q)
    
    # define x and y values
    x = curve[:,0]**2
    y = np.log(curve[:,1])
    
    # mask x and y data to low q values only 
    x_masked = x[:np.max(np.where(x < 0.0025))]
    y_masked = y[:np.max(np.where(x < 0.0025))]

    # Perform the curve fit using the Guinier equation
    popt, pcov = scipy.optimize.curve_fit(linear, x_masked, y_masked, method='lm', p0=[10,10], maxfev=50000)

    # Extract the fitted parameters
    I_0 = np.exp(popt[1])
    slope = popt[0]
    I_0_error = np.sqrt(pcov[1][1])
    I_0_error_scaled = I_0 * I_0_error
    Rg = np.sqrt(3 * slope)
    Rg_error = np.sqrt(pcov[0][0])
    Rg_error_scaled = 0.5 * Rg * ( Rg_error / abs(slope) )
    
    # Print values
    print("Rg  = {:.2f} +/- {:.2f}".format(Rg,Rg_error_scaled))
    print("I_0 = {:.2f} +/- {:.2f}".format(I_0,I_0_error_scaled))
    print('\n')
    
    # fit model to data
    print('Fitting model...')
    model = linear(np.linspace(np.min(x_masked), np.max(x_masked), len(x_masked)), *popt)
    
    # plot data
    print('Plotting Data...')
    
    # get number of x values
    xmax = np.max(np.where(x_masked == np.max(x_masked))) + 15
    
    # set up axis
    ax1 = plt.axes([0.125,0.125, 5, 5])
    
    # make plot 
    plt.scatter(x[:xmax], y[:xmax], label='Data', s=500, color='grey', zorder=1)
    plt.plot(x_masked, model, label='Linear Fit', linewidth=10, color='red', zorder=-1)
    
    # style plot
    plt.xlabel('q$^2$', fontsize=60, fontweight='bold')
    plt.ylabel('q$^2$I', fontsize=60, fontweight='bold')
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.title(str(label) + ' Guinier Analysis', fontsize=70, fontweight='bold')
    plt.legend(fontsize=60)
    plt.text(0.5, 0.05,'R$_{g}$ = ' + "{:.5f}".format(Rg) + ' ± ' + "{:.5f}".format((Rg_error_scaled)) + '\nI$_{0}$ = ' + "{:.5f}".format(I_0) + ' ± ' + "{:.5f}".format((I_0_error_scaled)),
             horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=70, color='red')
    
    # set thickness of graph borders
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(5)
        
    # define inset plot
    ax2 = plt.axes([-5, 0.5, 4, 4])
    
    # make plot 
    plt.scatter(x[:xmax], y[:xmax], label='Data', s=500, color='grey', zorder=1)
    plt.plot(x_masked, model, label='Linear Fit', linewidth=10, color='red', zorder=-1)
    
    # style plot
    plt.xlabel('q$^2$', fontsize=60, fontweight='bold')
    plt.ylabel('q$^2$I', fontsize=60, fontweight='bold')
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.title(str(label) + ' Guinier Analysis', fontsize=70, fontweight='bold')
    plt.legend(fontsize=60)
    plt.xlim(0, qmax)
    plt.text(0.5, 0.05,'R$_{g}$ = ' + "{:.5f}".format(Rg) + ' ± ' + "{:.5f}".format((Rg_error_scaled)) + '\nI$_{0}$ = ' + "{:.5f}".format(I_0) + ' ± ' + "{:.5f}".format((I_0_error_scaled)),
             horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=70, color='red')
    
    # set thickness of graph borders
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(5)
    
    # mark inset
    mark_inset(ax1, ax2, loc1=1, loc2=4, fc="none", ec="0.5", 
                linewidth=4)
    
    
    plt.show()


    return Rg, Rg_error_scaled, I_0, I_0_error_scaled

