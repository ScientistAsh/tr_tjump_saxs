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

import numpy as np
from scipy.optimize import curve_fit
import datetime

import numpy as np
from scipy.optimize import curve_fit

def guinier_analysis(file, label, delim=',', mask=0, qmax=0.0025, outdir=None):
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
            
        outdir (optional) : str
            Path to store output files. Output Rg and I will be saved as 
            well as the fitted model. If set to None, no files will be
            saved. The default value is None.

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
            
        model : np.array
            Linear fit of the Guinier analysis
            
    Rasies:
    -------
    
    Examnples: 
    ----------
    > temps = ['30C', '35C', '40C', '44C']
    
    > static_files = ['/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/ANALYSIS/STATIC/BUFFER_SUB/30C_buffsub.csv',
                     '/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/ANALYSIS/STATIC/BUFFER_SUB/35C_buffsub.csv',
                     '/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/ANALYSIS/STATIC/BUFFER_SUB/40C_buffsub.csv',
                     '/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/ANALYSIS/STATIC/BUFFER_SUB/44C_buffsub.csv']
                     
    > for t, f in zip(temps, static_files):
        print('Running Guinier Analysis on ' + str(t))
        Rg, Rg_error_scaled, I_0, I_0_error_scaled, model = guinier_analysis(file=str(f), label=str(t), delim=',', 
                                                                            mask=0, qmax=0.0025, 
                                                                            outdir='/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/ANALYSIS/STATIC/GUINIER/')
    '''
    
    # load data 
    print('Loading data...')
    curve = load_saxs(file=file, delim=delim, mask=mask)
    
    # define guinier equation
    def linear(x,a,b):
        return b-a*x
    
    # select qmin and qmax
    print('Running Guinier Analysis...')
    qmin = np.min(curve[:, 0])
    
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
    #plt.yscale('log')
    
    # style plot
    plt.xlabel('q$^2$', fontsize=60, fontweight='bold')
    plt.ylabel('ln(I)', fontsize=60, fontweight='bold')
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
    #plt.yscale('log')
    
    # style plot
    plt.xlabel('q$^2$', fontsize=60, fontweight='bold')
    plt.ylabel('ln(I)', fontsize=60, fontweight='bold')
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
    
    # check if files should be saved
    if outdir is not None:
        # get current data
        current_date = datetime.datetime.now().strftime('%d %B %Y')
        
        # make output directory
        make_dir(f=str(outdir))
    
        # save guinier analysis
        np.savetxt(str(outdir + label) + '_guinier.csv', np.c_[x[:xmax], y[:xmax]], header='q^2,ln(I)',
                   delimiter=',', comments='# Guinier Analysis for ' + str(label) + ' | DHVI | Henderson Lab | ALB | ' + str(current_date))
        
        np.savetxt(str(outdir + label) + '_guinier_params.csv', np.c_[Rg, Rg_error_scaled, I_0, I_0_error_scaled], header='Rg,Rg_err,I0,I0_err',
                   delimiter=',', comments='# Guinier Rg and I0 parameters for ' + str(label) + ' | DHVI | Henderson Lab | ALB | ' + str(current_date))
    
        np.savetxt(str(outdir + label) + '_guinier_fit.csv', np.c_[x_masked, model], header='q2,ln(I)',
                   delimiter=',', comments='# Guinier Fit for ' + str(label) + ' | DHVI | Henderson Lab | ALB | ' + str(current_date))
    
    
    
    plt.show()


    return Rg, Rg_error_scaled, I_0, I_0_error_scaled, model




def sys_err(flist, outlier_files, fslice=None, bin_size=10, 
            x='scattering vector (q, $\AA^-1$)', 
            y='$\Delta$ scattering intensity (i)', 
            title='CH505TF SAXS Scattering', save=True, save_dir='./', 
            save_name='sys_err_qc.png'):
    '''
    Description:
    ------------
    Function that will take an input file list, bin the files, average
    each bin, and compare the average of the bins to check for systemic
    errors in data collection. Before binning anf averaging, the function
    will automatically remove pre-determined outlier files. Outliers 
    should be previoulsy determined and can be determined with the 
    `svd_outliers` and/or `iterative_chi` functions. This function is 
    intended to be used on scattering or difference curve sets, with each 
    scattering curve image imported into the file list. Funtion will plot
    the mean of each bin, save the plot, and return the cleaned flist, bins,
    and the mean of each bin. 
    
    Parameters:
    -----------
    flist : list
        List of files to run systemic error analysis on. A file list can
        be made with the `make_flist` function of the CH505TF_SAXS module. 
        
    outlier_files : list
        List of files containing outliers images. The outliers can be determined
        with the `svd_outliers` and `iterative_chi` functions. These images will
        be removed from the flist before binning and averaging.
        
    fslice (optional) : list
        List containing integers that will slice the replica 
        number from the file name. If set to None, then the 
        entire file name will be used. The default value is 
        None. 
    
    bin_size (optional) : int
        Number of images to include in each bin. Default value is 10. 
        
    x (optional) : str
        Label for x axis of plot. Default value is 'scattering vector (q, $\AA^-1$)'
        
    y (optional) : str
        Label of plot y axis. Default value is '$\Delta$ scattering intensity (i)'
    
    title (optional) : str
        Title for plot. Default value is 'SAXS Scattering'
        
    save (optional) : bool
        Indicates if the plot should be saved to file. If set to False 
        then the plot will not be saved. Default value is False.
        
    save_dir (optional) : str
        Directory to store output files in. If the directory does not exist, then 
        it will be made. The default value is None. 
        working directory. 
        
    save_name (optional) : str
        File name to save plot to. Plot only will be saved if save=True. 
        
    
    Returns:
    --------
    flist : list
        List containing the files with the outliers removed.
        
    bins : list
        List of bins used for analysis.
    
    means : list
        List of means of each bin 
    
    '''

    # remove outliers
    for o in outlier_files:
        flist = remove_outliers(flist=flist, outlier_file=o, fslice=[-9,-6])
        
    # sort files so they are in order that they were collected
    flist.sort()
    
    # load scattering curves as a set
    data, data_arr, q, err = load_set(flist=flist, delim=' ', mask=10, err=False)
    
    # create bins for list
    bins = []
    for i in range(0, len(data_arr), bin_size):
        bin = data_arr[i: i+10]
        bins.append(bin)
    
    # calculate average curve of each bin
    means = []
    for bin in bins:
        avg = bin.mean(axis=0)
        means.append(avg)
        
    # create a list of labels
    labels = list(range(1, len(bins) +1))
        
    # plot each bin average
    plot_curve(data_arr=means, q_arr=q, labels=labels, qmin=0.015, qmax=0.1, imin=None, imax=None, x=x, y=y, 
               title=title, save=save, save_dir=save_dir, 
               save_name=save_name)
       
    
    
    
    return flist, bins, means


def kratky_plot(files, delim=',', mask=0, err=True, labels=None, qmin=None, qmax=None, imin=None, 
                imax=None, x='scattering vector (Å$^{-1}$)', y='q$^{2}$I', title='SAXS Scattering Curve', 
                save=False, save_dir=None, save_name=None):
    '''
    Description:
    ------------
    Function to calculate and save a Kratky plot for a set of input curves.
    Function will also save the Kratky data in a csv file. 
    
    Parameters:
    -----------
    files : list
        List contining files, with full path, containing SAXS curves to make
        Kratky plots for. Assumes that the input files 
        contain two columns with the scattering vector (q) and the 
        scattering intensity (i). If err=True, the the experimental error will
        also be loaded as long as the input data file contains a third column with 
        the experimental error. 
        
    delim (optional) : str
        Delimitter used in input files. Default valaue is comma (','). 
        
    mask (optional) : int
        Number of rows to skip when loading files. Default values is 0. Useful for
        skipping rows with NaN or masked values. 
        
    err (optional) : bool
        Indicates if there is the column containing experimental error. If set 
        to False, then no errors will be loaded. Default value is False. 

    labels (optional) : list
        List containing labels to use for plot. If set to None, then output plot 
        will have no labels. Default value is None. 
        
    qmin (optional) : float
        Minimum Q value for a zoomed in inset plot. If set to None,
        then the qmin will be determined from the minimum q value in
        the input set of curves. If both qmin and qmax are set to
        None, then the inset will include the entire Q range. If qmin,
        qmax, imin, and imax are all set to None, then no inset plot 
        will be made. Default value is None. 
        
    qmax (optional) : float
        Maximum Q value for a zoomed in inset plot. If set to None,
        then the qmax will be determined from the maximum q value in
        the input set of curves. If both qmin and qmax are set to
        None, then the inset will include the entire Q range. If qmin,
        qmax, imin, and imax are all set to None, then no inset plot 
        will be made. Default value is None. 
        
    imin (optional) : float
        Minimum I value for a zoomed in inset plot. If set to None,
        then the imin will be determined from the minimum I value in
        the input set of curves. If both imin and imax are set to
        None, then the inset will include the entire I range. If qmin,
        qmax, imin, and imax are all set to None, then no inset plot 
        will be made. Default value is None. 
        
    imax (optional) : float
        Maximum I value for a zoomed in inset plot. If set to None,
        then the imax will be determined from the maximum I value in
        the input set of curves. If both imin and imax are set to
        None, then the inset will include the entire I range. If qmin,
        qmax, imin, and imax are all set to None, then no inset plot 
        will be made. Default value is None.
        
    x (optional) : str
        Label for x axis of plot. Default value is 'scattering vector'
        
    y (optional) : str
        Label of plot y axis. Default value is 'q$^{2}$I'
    
    title (optional) : str
        Title for plot. Default value is 'SAXS Scattering Curve'
        
    save (optional) : bool
        Indicates if the plot should be saved to file. If set to False 
        then the plot will not be saved. Default value is False.
        
    save_dir (optional) : str
        Directory to store output files in. If the directory does not exist, then 
        it will be made. The default value is None. 
        working directory. 
        
    save_name (optional) : str
        File name to save plot to. Plot only will be saved if save=True. 
        
    Returns:
    --------
    kratky : np.array
        Numpy array containing Kratky values (without q).
    
    '''
    data, data_arr, q, err = load_set(flist=files, delim=delim, mask=mask, err=err)
    
    # get kratky array
    kratky = data_arr * (q**2)

    
    # plot Kratky curves
    plot_curve(data_arr=kratky, q_arr=q, labels=labels, qmin=qmin, qmax=qmax, imin=imin, 
               imax=imax, x=x, y=y, title=title, save=save, save_dir=save_dir, save_name=save_name)
    
    # save kratky data
    if save is True:
        make_dir(f=save_dir)
        save_data = np.vstack([q, kratky])
        np.savetxt(str(save_dir) + '/' + str(save_name) + '_kratky.csv', 
                   np.c_[save_data], delimiter=",")
    
    return kratky

def temp_cal(flist, temps, test, delim=',', test_delim=',', mask=0, test_mask=0, err=False, outdir=None, outfile=None, xlab='Temperature (°C)',
            ylab='Maximum Scattering Intensity', plot_title='T-Jump Calibration\nLinear Regression'):
    '''
    Description:
    ------------
    Function that will calculate the maximum scattering intensity 
    for the scattering curve in the indicated input file. This 
    function is intended for use to calibrate temperature jumps
    for time resolved, temperature-jump SAXS data. Function fits
    the static SAXS at the input temperatures to a linear regression
    function based on np.polyfit() function for the calibration. This 
    function will return the the r2 score. If outfile is set to True
    then CSV files of the imax and error (is err=True) and a separate 
    file the the fitted model parameters will be saved along with a plot 
    of the data with the fit in the indicated outdir directory. If the 
    indicated outdir does not exist, it will be created. 
    
    Parameters:
    -----------
    flist : list
        List of files to use for T-Jump calibration. These should be static SAXS
        scattering curves at at least 3 different temperatures. 
    
    temps : list
        List of temperatures corresponding to the input scattering curves. 
        
    test : str
        File, including full path, containing the water T-Jump test difference curve
        
    delim (optional) : str
        Delimitter used in data files. Default is comma. 
        
    test_delim (optional) : str
        Delimitter used in test T-Jump files. Default is comma. 
        
    mask (optional) : int
        Number of rows to skip when loading files. Default values is 0. Useful for
        skipping rows with NaN or masked values. 
        
    test_mask (optional) : int
        Number of rows to skip when T-jump test file. Default values is 0. Useful for
        skipping rows with NaN or masked values. 
        
    err (optional) : bool
        Indicates if there is the column containing experimental error. If set 
        to False, then no errors will be loaded. Default value is False. 
        
            
    outfile (optional) : str
        Prefix to use for file names. CSV files containing the fitted curve will be saved with 
        the '.csv' suffix and plot files will be saved with the '.png' suffix. Default value is 
        'fit'. 
        
    outdir (optional) : str
        Location to save plot and modeled curve. If set to None then no CSV or PNG plot files
        will be saved. When set to None, no files will be saved. Default value is None. 
    
    xlab (optional) : str
        Label to use for X-axis in plot. Default value is 'Temperature (°C).'
    
    ylab (optional) : str
        Label to us for Y-axis in plot. Default value is 'Maximum Scattering Intensity'

    
    plot_title (optional) : str
        Label to use for title of plot. Default value is 'T-Jump Calibration\nLinear Regression'
    
    Returns:
    --------
    data_arr : np.array
        Array containing the input scattering curves. 
        
    imax : list
        List containing the imax values for each input scattering curve. 
        
    model : np.array
        Array containing the fitted parameters for the linear regression model.
        
    r2 : float
        Correlation coefficient for the linear regression fit. 
        
    imax_err : np.array
        Array containing error associated with the max_i values from each temp. 
        
    p : float
        Predicted temperature for input T-jump test.
        
    Raises:
    -------
    ValueError 
        If err parameter is something other than True or False. 
    '''
       
    # load file
    data, data_arr, q, error = load_set(flist=flist, delim=delim, mask=mask, err=err)
    
    # Load Test T-Jump
    jump_test = load_saxs(file=str(test), delim=test_delim, mask=test_mask)
    test_imax = jump_test[:, 1].max()
        
    # create list for storing imax values
    imax = [] 
    imax_err = []
    
    # Make outdir
    make_dir(f=str(outdir))
    # Determine max scattering
    if err is False:
        for d, t in zip(data_arr, temps):
            max_i = d.max()
            print(str(t) + ' Max I = ' + str(max_i))
        
            # Append imax to list
            imax.append(max_i)
            
        # Write max_i to files
        np.savetxt(str(outdir) + 'imax_' + str(outfile) + '.csv', np.c_[temps, imax], delimiter=",", header='temp, max_i')
    
    # Determine SEM
    elif err is True:
        for t, e, d in zip(temps, error, data_arr):
            max_i = d.max()
            imax.append(max_i)
            n = np.where(d == max_i)
            max_err = e[n]
            imax_err.append(max_err[0])
            print(str(t) + '°C Max. I +/- SEM = ' + str(max_i) + ' +/- ' + str(max_err[0]))
        
        x = np.where(jump_test[:,1] == test_imax.max())
        test_err = jump_test[x, 2]
            
        # Write imax and sem to file
        np.savetxt(str(outdir) + 'imax_' + str(outfile) + '.csv', np.c_[temps, imax, imax_err], delimiter=",", header='temp, max_i, sem')
        
    else:
        raise ValueError('err parameter must be either True or False')
          
            
    # Fit imax to linear regression
    model = np.polyfit(temps, imax, deg=1, full=False)
    predict = np.poly1d(model)
    r2 = r2_score(predict(temps), imax)
    
    # Predict temperature for test t-jump
    p = (test_imax - model[1]) / model[0]
    
    # Fit Data to Model
    m = np.array([range(0, max(temps) + 4), predict(range(0, max(temps) + 4))])
    np.savetxt(str(outdir) + 'fit_' + str(outfile) + '.csv', np.c_[m[0], m[1]], delimiter=",", header='temp, max_i')
    
    # Write model and r2 to file
    np.savetxt(str(outdir) + 'model_' + str(outfile) + '.csv', np.c_[model[0], model[1],  r2, p, test_imax, test_err[0][0]], delimiter=",", 
               header='m, b, r2, temp_predict, tjump_imax, predict_err')
    
    # plot imax + fit
    ax = plt.axes([0.125,0.125, 5, 5])
    plt.plot(m[0], m[1], linewidth=10, color='red', label='Linear Regression',zorder=-1)
    if err is False: 
        plt.scatter(temps, imax, marker='o', s=500., color='grey', label='Data', zorder=1)
        plt.scatter(p, test_imax, color='blue', marker='*', s=1000, label='H$_{2}$O T-Jump', zorder=1)
        ax.annotate('T-Jump Test Temp. = ' + "{:.5f}".format(p) + '°C', (p, test_imax), fontsize=50, color='blue')
    if err is True:
        plt.errorbar(temps, imax, yerr=imax_err, fmt='o', ms = 20., color='grey', label='Data', linewidth=5, capthick=5)
        plt.errorbar(p, test_imax, yerr=test_err[0][0], color='blue', fmt='*', ms=40, label='T-Jump', zorder=1)
        ax.annotate('T-Jump Test Temp. = ' + "{:.5f}".format(p) + '°C', (p, test_imax), fontsize=50, color='blue')
    plt.xlabel(str(xlab), fontsize=60, fontweight='bold')
    plt.ylabel(str(ylab), fontsize=60, fontweight='bold')
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.title(str(plot_title), fontsize=70, fontweight='bold')
    plt.text(0.5, 0.05,'y = ' + "{:.5f}".format(model[0]) + 'x + ' + "{:.5f}".format((model[1])) + '\nr$^{2}$ = ' + "{:.5f}".format(r2), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=70, color='red')
    plt.legend(fontsize=60)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(5)

    # save figure
    if outfile is not None:
        plt.savefig(str(outdir) + 'plot_' + str(outfile) + '.png', bbox_inches='tight')

    plt.show()

    
    return data_arr, imax, model, r2, imax_err, p, #test_err[0][0]