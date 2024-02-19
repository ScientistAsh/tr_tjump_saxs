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
import datetime

def saxs_auc(flist, times=[1.5, 3, 5, 10, 50, 100, 300, 500, 1000],
             delim=',', mask=0, qmin=None, qmax=None, outdir=None,
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
        
    mask (optional) : int
        Number of rows to skip when loading files. Helpful when a mask was
        applied during data collection and low scattering vector values should
        not be analyzed. When set to 0 all rows are loaded. Default value 
        is 0. 
    
    q_min : float
        Q value to begin integration at. If set to None, the will start at lowest
        Q value in input data set. Default value is None. 
        
    q_max : float
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
        
    Examples:
    ---------
        
    ex.1: Continue using data (data will also be saved):
    a = saxs_auc(flist=files, delim=',', q_min=0, 
                q_max=0.3, 
                outdir='../../ANALYSIS/NALYSIS/AUC/', 
                data_outfile='tjumps_auc.csv',
                plot_outfile='tjump_auc_plot.png')
                
    ex.2: Just save data to csv file:
    saxs_auc(flist=files, delim=',', q_min=0, q_max=0.3)
    '''
    
    # set variable names
    auc = []   
    data = []
    
    # loop over difference files for time_delay in directory
    for t, f in zip(times, flist):
        
        print('\033[94;1mLoading ' + str(f) + '\033[94;1m')

        # convert time to integer
        #if t[-2:] == 'ms':
        #    t = int(t[:-2]) * 1000
        #else:
        #    t = int(t[:-2])
            
        # load data
        data = load_saxs(file=f, delim=delim, mask=mask)
        
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
        np.savetxt(str(outdir) + str(data_outfile), np.c_[auc[:, 0], auc[:, 1]], delimiter=',',
                  header='time_delay (us),auc', comments='# AUC Analysis | DHVI | Henderson Lab | ALB | ' + datetime.datetime.now().strftime('%d %B %Y'))
 
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
   
    return auc



def svd_kinectics(flist, delim=',', times=[1.5,3,5,10,50,100,300,500,1000], 
                  time_unit='us', outdir=None):
    '''
    Function to perform SVD on SAXS curves to identify signal
    components. Saves the right and left vectors as a CSV file.
    
    Parameters:
    -----------
    flist : list
        List contining files, with full path, containing data to run SVD on. 
        Assumes that the input files contain 
        
    times (optional) : list
        List of times included in SVD. Default value is 10, 50, 100, 500, and 100 
        microseconds. 
        
    time_unit (optional) : str
        Unit for times included in SVD. Default value is us for microseconds. 
        
    outdir (optional) : str
        Name of directory to store output plots to. A scree plot is output
        along with output plots of both the first two left vectors and the
        first two right vectors will be made, with the left and right vectors
        in separate plots. CSV files of the right and left vectors are also saved. 
        If set to None, then no outfile will be saved. Default value is None. 
        
    Returns:
    -------- 
    u :  array. 
        The first a.ndim - 2 dimensions have the same size as those 
        of the input.
        
    s : array
        Vector with the singular values, within each vector sorted in 
        descending order. The first a.ndim - 2 dimensions have the same size
        as those of the input.
        
    v : array
        Unitary array. The first a.ndim - 2 dimensions have the same size as 
        those of the input. 
    
    Examples:
    ---------
    u, s, v = svd_kinectics(flist=cleaned_files, delim=' ', times=[10, 50, 100, 25s, 50s, 75s, 1000], time_unit='us', 
                        outdir='./ANALYSIS/MODELS/SVD/')
    '''
    # create empty vectors list
    vectors = []
    
    # load data
    print('Loading Data...')

    # load data
    print('Loading Data...')
    for f in flist:
        curve = load_saxs(file=f, delim=delim, mask=0)

        # append curve to vector list
        vectors.append((curve[:,1]))
    # get q
    q = curve[:,0]
    
    # svd calculation
    print('Running SVD...')
    matrix = np.matrix([i for i in vectors]).transpose()    
    u,s,v = svd(matrix, full_matrices=False)
    
    # make scree plot
    eigvals = s**2 / np.sum(s**2)  
    ax1 = plt.axes([0.125,0.125, 5, 5])
    sing_vals = np.arange(len(flist))
    plt.plot(sing_vals, eigvals, 'ro-', linewidth=5)
    plt.title('Scree Plot', fontsize=70)
    plt.xlabel('Components', fontsize=60)
    plt.ylabel('Eigenvalue', fontsize=60)
    plt.xticks(fontsize=55)
    plt.yticks(fontsize=55)
    plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, 
               shadow=False, fontsize=60,
               markerscale=0.4)
    
    # save scree plot
    if outdir is not None:
        make_dir(str(outdir) + '/PLOTS/')
        outfile = 'scree.png'
        plt.savefig(str(outdir + '/PLOTS/' + outfile), bbox_inches='tight')
    
    # show plot    
    plt.show()

        
    # plot first two left vectors
    plt.clf()
    ax2 = plt.axes([0.125,0.125, 5, 5])
    plt.plot(q, (u[:,0]), label='LV1', linewidth=5)
    plt.plot(q, (u[:,1]), label='LV2', linewidth=5)
    plt.title('SVD Left Vectors', fontsize=70)
    plt.xlabel('Q (Å' + r'$^{-1}$' + ')', fontsize=60)
    plt.ylabel('L Vectors', fontsize=60)
    plt.xticks(fontsize=55)
    plt.yticks(fontsize=55)
    plt.xlim([0.0, 1.0])
    plt.legend()
    
    # save left vector plot
    if outdir is not None:
        outfile = 'left_vectors.png'
        plt.savefig(str(outdir + '/PLOTS/' + outfile), bbox_inches='tight')
        
    # show plot
    plt.show()
    
    # convert u and v to array
    lu = np.asarray(u)
    lv = np.asarray(v)
    timer = np.asarray(times)
    
    # plot first 2 right vectors
    ax3 = plt.axes([0.125,0.125, 5, 5])
    
    plt.plot(timer, lv[0], label='RV1', linewidth=5)
    plt.plot(timer, lv[1], label='RV2', linewidth=5)
    plt.xlabel('time (' + r'$\mu$' + 's)', fontsize=60)
    plt.title('SVD Right Vectors', fontsize=70)
    plt.ylabel('R Vectors', fontsize=60)
    plt.xticks(fontsize=55)
    plt.yticks(fontsize=55)
    plt.legend(loc='best', fontsize=60)
    
    # save right vector plot
    if outdir is not None:
        outfile = 'Rvectors.png'
        plt.savefig(str(outdir + '/PLOTS/' + outfile), bbox_inches='tight')
     
    # show plot
    plt.show()
    
    # save right and left vectors as csv
    np.savetxt(str(outdir) + 'right_vectors.csv', 
               lv, delimiter=",")
    np.savetxt(str(outdir) + 'left_vectors.csv', 
               lu, delimiter=",")
    
    return u, s, v


def auc_fit(file, x, columns=None, delim=',', skip=0, func='double', initial_guess=[1, 1, 1],
            iterations=2000, xlab='Time Delay $\mu$s', ylab='Area Under the Curve\n(Simpsons Rule)',
            xlogscale=True, ylogscale=False, outdir=None, outfile='fit',
            plot_title='CH505TF SAXS T-Jumps Area Under the Curve\nExponential Fit'):
    '''
    Description:
    ------------
    Function to fit curve to either an exponential decay or double exponential
    decay. The single exponential decay function takes the form:
                        A * np.exp(-x/tau) + B
                        
    and the the double exponential decay funtion takes the form:
            A1 * np.exp(-x/tau1) + A2 * np.exp(-x/tau2) + B
            
    The function will return the fit and plot the fit overlayed with input data. 
    Function is based on the scipy.optimize.curve_fit() function and returns the values
    for the fitted parameters. 
    
    Parameters:
    -----------
    file : str
        String containing the file name, with full path, containing ydata to be fit. 
        
    x : np.array
        Numpy array containing the X-data. 
        
    columns (optional) : int or list of int
        Columns to be loaded from the input file. Columns are 0 indexed. If set to
        None, then all columns will be loaded. Default value is None. 
        
    skip (optional) : int
        Row's to skip. Data is loaded as a numpy array so headers cannot be loaded.
        skip=0 then all rows will be loaded. Default value is 0.   
    
    func (optional) : str
        What type of function to fit to data. Accepts either 'double' for double 
        exponential decay fit or 'single' for single exponential decay fit. Will raise
        and ValueError for incorrect value. 
        
    initial_guess (optional) : array_like
        Initial guess for the parameters. The default sets all the initial values to 1.
        
    iterations (optional) : int
        Number of iterations to run the scipy.optimize.curve_fit. Default value is 2000. 
        
            
    xlab (optional) : str
        Label to use for X-axis in plot. Default value is 'Time Delay (us).'
    
    ylab (optional) : str
        Label to us for Y-axis in plot. Default value is 'Area Under the Curve\nSimpsons Rule'
        
    xlogscale (optional) : bool
        Boolean to indicate if a log scale should be used on the x axis. If set to False then 
        the absolute values will be used. Default value is True.
        
    ylogscale (optional) : bool
        Boolean to indicate if a log scale should be used on the y-axis. If set to False then
        the absolute values will be used. Default value is False. 
        
    outdir (optional) : str
        Location to save plot and modeled curve. If set to None then no CSV or PNG plot files
        will be saved. Default value is None. 
        
    outfile (optional) : str
        Prefix to use for file names. CSV files containing the fitted curve will be saved with 
        the '.csv' suffix and plot files will be saved with the '.png' suffix. Default value is 
        'fit'. 
    
    plot_title (optional) : str
        Label to use for title of plot. Default value is 'CH505TF SAXS T-Jumps Area Under the Curve'
        
        
    Returns:
    --------
    
    model : np.array
        Model of fit. 
        
    popt : np.array
        Values of funcotional parameters determined from exponential fit. 
        
    conf_int : np.array
        Confidence intervals for the fitted parameters. 
        
    Examples:
    ---------
    auc_fit(file='/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/ANALYSIS/Trimer10.17_Series3/BOOTSTRAPPING/AUC/bootstrap_auc_resamples.csv', x=[5,10,50,100,500,1000], columns=[1,2],
       delim=',', skip=0, func='double', initial_guess=[10.0, 1.0, 25.0, 1.0, 500.0], iterations=2000, xlab='Time Delay $\\mu$s', ylab='AUC', xlogscale=True, ylogscale=False, 
       outdir=None, outfile='test', plot_title='CH848 TrimerOnly Series3 AUC')
        
    '''
    
    # Define single exponential decay function
    def single_exp(x, A, tau, B):
        return A * np.exp(-x/tau) + B
    
    # define double exponential decay function
    def double_exp(x, A1, tau1, A2, tau2, B):
        return A1 * np.exp(-x/tau1) + A2 * np.exp(-x/tau2) + B

    # load data
    ydata = np.loadtxt(fname=file, usecols=columns, delimiter=delim, skiprows=skip)[:, 1] 
    
    # define x data
    xdata = x
    
    # fit data to exponential function
    if func == 'single':
        popt, pcov = scipy.optimize.curve_fit(single_exp, xdata, ydata, maxfev=iterations, p0=initial_guess)
        model = single_exp(np.linspace(min(xdata), max(xdata), int(max(xdata))), *popt)
        params =['A', 'tau', 'B']
        
    elif func == 'double':
        popt, pcov = scipy.optimize.curve_fit(double_exp, xdata, ydata, maxfev=iterations, p0=initial_guess)
        model = double_exp(np.linspace(min(xdata), max(xdata), int(max(xdata))), *popt)
        params = ['A1', 'tau1', 'A2', 'tau2', 'B']
   
    else:
        raise ValueError('Invalid function type. Expected one of: single, double')
        
    alpha = 0.05  # 95% confidence interval = 100*(1-alpha)

    n = len(ydata)    # number of data points
    p = len(popt)  # number of parameters

    dof = max(0, n - p)  # number of degrees of freedom

    # student-t value for the dof and confidence level
    t_val = scipy.stats.t.ppf(1.0-alpha/2., dof) 

    sigma = np.diag(pcov)**0.5
    conf_int = t_val * sigma/np.sqrt(n)

    if func == 'single':
        print("Single Exponential Fit:")
        print("A =", popt[0], "±", conf_int[0])
        print("tau =", popt[1], "±", conf_int[1])
        print("B =", popt[2], "±", conf_int[2])

    elif func == 'double':
        print("\nDouble Exponential Fit:")
        print("A1 =", popt[0], "±", conf_int[0])
        print("tau1 =", popt[1], "±", conf_int[1])
        print("A2 =", popt[2], "±", conf_int[2])
        print("tau2 =", popt[3], "±", conf_int[3])
        print("B =", popt[4], "±", conf_int[4])
       

    # plot data
    ax = plt.axes([0.125,0.125, 5, 5])
    plt.scatter(xdata, ydata, marker='o', s=500, label='data', color='grey')
    plt.plot(model, linestyle='-', color='red', label='fit')
    plt.xlabel(xlab, fontsize=50, fontweight='bold')
    plt.ylabel(ylab, fontsize=50, fontweight='bold')
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)
    plt.title(plot_title, fontsize=70, fontweight='bold')
    if xlogscale is True:
        plt.xscale('log')
    if ylogscale is True:
        plt.yscale('log')
    plt.legend(loc='best', borderpad=0.3, shadow=False, fontsize=60,
               markerscale=1)

        
    # save data
    if outdir is not None:
        make_dir(f=outdir)
        plt.savefig(str(outdir) + '/' + str(outfile) + '.png', bbox_inches='tight')

        np.savetxt(outdir + outfile + '_' + str(func) + '_model.csv', 
                   np.c_[np.linspace(min(xdata), max(xdata), int(max(xdata))), model], 
                   delimiter=',', header='time,' + str(func) + '_exp_decay_model',
                   comments='# ' + str(plot_title) + ' | Calculated with tr_tjump_saxs_analysis Python package | ' + datetime.datetime.now().strftime('%d %B %Y'))

        np.savetxt(outdir + outfile + '_' + str(func) + '_fit.csv',
                   [popt, conf_int], delimiter=',', header=str(params))
        
    plt.show()
        
    return model, popt, conf_int


def svd_fit(file, x, delim=',', row=0, func='double', initial_guess=[1, 1, 1], 
            iterations=2000, xlab='Time Delay $\mu$s', ylab='% Contribution',
            xlogscale=True, ylogscale=False, outdir=None, outfile='fit',
            plot_title='CH505TF SAXS T-Jumps First Right Vector\nExponential Fit'):
    '''
    Description:
    ------------
    Function to fit curve to either an exponential decay or double exponential
    decay. The single exponential decay function takes the form:
                        A * np.exp(-x/tau) + B
                        
    and the the double exponential decay funtion takes the form:
            A1 * np.exp(-x/tau1) + A2 * np.exp(-x/tau2) + B
            
    The function will return the fit and plot the fit overlayed with input data. 
    Function is based on the scipy.optimize.curve_fit() function and returns the values
    for the fitted parameters. 
    
    Parameters:
    -----------
    file : str
        String containing the file name, with full path, containing ydata to be fit. 
        
    x : np.array
        Numpy array containing the X-data. 
        
    delim (optional) : str
        Delimitter used in file to load. Default value is comma. 
        
    row (optional) : int
        Row containing right vector to fit. Rows are 0 indexed. Default value is 0, 
        which corresponds to the first right vector. 
    
    func (optional) : str
        What type of function to fit to data. Accepts either 'double' for double 
        exponential decay fit or 'single' for single exponential decay fit. Will raise
        and ValueError for incorrect value. 
        
    initial_guess (optional) : array_like
        Initial guess for the parameters. The default sets all the initial values to 1.
        
    iterations (optional) : int
        Number of iterations to run the scipy.optimize.curve_fit. Default value is 2000. 
        
            
    xlab (optional) : str
        Label to use for X-axis in plot. Default value is 'Time Delay (us).'
    
    ylab (optional) : str
        Label to us for Y-axis in plot. Default value is '% Contribution'
        
    xlogscale (optional) : bool
        Boolean to indicate if a log scale should be used on the x axis. If set to False then 
        the absolute values will be used. Default value is True.
        
    ylogscale (optional) : bool
        Boolean to indicate if a log scale should be used on the y-axis. If set to False then
        the absolute values will be used. Default value is False. 
        
    outdir (optional) : str
        Location to save plot and modeled curve. If set to None then no CSV or PNG plot files
        will be saved. Default value is None. 
        
    outfile (optional) : str
        Prefix to use for file names. CSV files containing the fitted curve will be saved with 
        the '.csv' suffix and plot files will be saved with the '.png' suffix. Default value is 
        'fit'. 
    
    plot_title (optional) : str
        Label to use for title of plot. Default value is 'CH505TF SAXS T-Jumps First Right Vector\n
        Exponential Fit'
        
        
    Returns:
    --------
    popt : np.array
        Values of funcotional parameters determined from exponential fit. 
            

    conf_int : np.array
        Confidence intervals for the fitted parameters. 
        
    model : np.array
        Model of fit. 

        
    Examples:
    ---------
    svd_fit(file='/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Vt_SVD.csv', x=[5,10,50,100,250, 500,750, 1000], row=0,
            delim=',', func='double', initial_guess=[10.0, 1.0, 25.0, 1.0, 500.0], iterations=2000, xlab='Time Delay $\\mu$s', ylab='Contribution',
            xlogscale=True, ylogscale=False, outdir=None, outfile='test', plot_title='CH848 TrimerOnly Series3 RV1')
        
    '''
    
    def single_exp(x, A, tau, B):
        return A * np.exp(-x/tau) + B
    
    # define double exponential decay function
    def double_exp(x, A1, tau1, A2, tau2, B):
        return A1 * np.exp(-x/tau1) + A2 * np.exp(-x/tau2) + B
    
    # load data
    data = np.loadtxt(fname=file, delimiter=delim) 
    ydata = data[row]
    
    # define x data
    xdata = np.array(x)
    
    # fit data to exponential function
    if func == 'single':
        popt, pcov = scipy.optimize.curve_fit(single_exp, xdata, ydata, maxfev=iterations, p0=initial_guess)
        model = single_exp(np.linspace(min(xdata), max(xdata), int(max(xdata))), *popt)
        tau = 1 / popt[1]
        params =['A', 'tau', 'B']
        
    elif func == 'double':
        popt, pcov = scipy.optimize.curve_fit(double_exp, xdata, ydata, maxfev=iterations, p0=initial_guess)
        model = double_exp(np.linspace(min(xdata), max(xdata), int(max(xdata))), *popt)
        tau_fast = 1 / popt[1]
        tau_slow = 1 / popt[3]
        params = ['A1', 'tau1', 'A2', 'tau2', 'B']
   
    else:
        raise ValueError('Invalid function type. Expected one of: single, double')
        
    alpha = 0.05  # 95% confidence interval = 100*(1-alpha)

    n = len(ydata)    # number of data points
    p = len(popt)  # number of parameters

    dof = max(0, n - p)  # number of degrees of freedom

    # student-t value for the dof and confidence level
    t_val = scipy.stats.t.ppf(1.0-alpha/2., dof) 

    sigma = np.diag(pcov)**0.5
    conf_int = t_val * sigma/np.sqrt(n)

    if func == 'single':
        print("Single Exponential Fit:")
        print("A =", popt[0], "±", conf_int[0])
        print("tau =", popt[1], "±", conf_int[1])
        print("B =", popt[2], "±", conf_int[2])

    elif func == 'double':
        print("\nDouble Exponential Fit:")
        print("A1 =", popt[0], "±", conf_int[0])
        print("tau1 =", popt[1], "±", conf_int[1])
        print("A2 =", popt[2], "±", conf_int[2])
        print("tau2 =", popt[3], "±", conf_int[3])
        print("B =", popt[4], "±", conf_int[4])
    
    # plot data
    ax = plt.axes([0.125,0.125, 5, 5])
    plt.scatter(xdata, ydata, marker='o', s=500, label='data', color='grey')
    plt.plot(model, linestyle='-', color='red', label='fit')
    plt.xlabel(xlab, fontsize=50, fontweight='bold')
    plt.ylabel(ylab, fontsize=50, fontweight='bold')
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)
    plt.title(plot_title, fontsize=70, fontweight='bold')
    if xlogscale is True:
        plt.xscale('log')
    if ylogscale is True:
        plt.yscale('log')
    plt.legend(loc='best', borderpad=0.3, shadow=False, fontsize=60,
               markerscale=1)

    # save figure
    if outdir is not None:
        make_dir(f=outdir)
        plt.savefig(str(outdir) + '/' + str(outfile) + '.png', bbox_inches='tight')
        
        np.savetxt(outdir + outfile + '_' + str(func) + '_model.csv', 
                   np.c_[np.linspace(min(xdata), max(xdata), int(max(xdata))), model], 
                   delimiter=',', header='time,' + str(func) + '_exp_decay_model',
                   comments='# ' + str(plot_title) + ' | Calculated with tr_tjump_saxs_analysis Python package | ' + datetime.datetime.now().strftime('%d %B %Y'))

        np.savetxt(outdir + outfile + '_' + str(func) + '_fit.csv',
                   [popt, conf_int], delimiter=',', header=str(params))
        
    plt.show()
        
    return popt, conf_int, model





