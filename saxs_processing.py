'''
tr_tjump_saxs: saxs_processing.py
Date created (mm-dd-yy): 04-11-2023

This module is part of the tr_tjump_saxs package for processing and anlayzing pump-probe time resolved, 
temperature-jump small angle X-ray scattering data sets. This module includes functions for detecting 
and removing outliers, scaling scattering/difference curves, and subtracting scattering/difference 
curves.

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


def svd_outliers(arr, flist, q, cutoff=2.5, save_dir='./OUTLIERS/',
                save_name='outliers.csv'):
    '''
    Function to run SVD outlier detection on a set of TR, T-Jump SAXS
    scattering curves. The function takes a 2D numpy array where each row is
    an individual scattering curve. The function also takes the associated file 
    lists as input. The function performs SVD on the input array, detecting outliers 
    using the first right vector according to the mean +/- the input threshold. This 
    function assumes that the order of curves in the input array matches the order of
    curves in the input file list. The plot of the first left vector is saved. The
    outliers are saved as a csv file to indicated directory with the indicated file 
    names. Returns the list of outlier files for the scattering curves. The 
    `remove_outlier` function can be used to remove the outliers from the file list. 
    
    Parameters:
    -----------
    arr : np.array
        Numpy array containing scattering curve data to run SVD outlier detection on. 
        
    flist : list
        List containing the files associated with the input scattering curves. This 
        list can be easily made with the `get_files` function. 
        
    q : np.array
        Numpy array containing the q values associated with the loaded scattering 
        curves. 
        
    cutoff (optional) : float
        Threshold used to determine outliers. Outliers are determined by finding
        curves where v1 is outside the mean of all v1 vectors by more than the 
        cutoff. The default value is 2.5. 
        
    save_dir (optional) : str
        Directory to store output outlier CSV files. The default value is a 
        subdirectory, 'OUTLIERS/' in the current working directory. If the directory
        does not already exist, it will be made. 
        
    save_name (optional) : str
        File name to store the list of scattering curve outliers. The default name is 
        outliers.csv.

    
    Returns:
    --------
    outlier_files : list
        List containing the files of cattering curve outliers. The outliers 
        can be removed from a file list with the `remove_outliers` function. 
        
    outliers : list
        List containing the index of scattering outlier curves. The outliers 
        can be removed from a file list with the `remove_outliers` function. 

        
    Examples:
    ----------
    outliers_1ms = svd_outliers(arr=data_arr, flist=unique_files, q=q, cutoff=2.5, save_dir='/datacommons/dhvi-md/AshleyB/tmp/', 
                               save_name='1ms_svd_outliers.csv')
    outliers_1ms
    > (['/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Trimer10.17Only_Series3_44C/processed_series3_44C_all_images/TrimerOnly_Series-3_44C_1ms_042_Q.chi'],
       [5])
     
    outliers_1ms[0]
    > ['/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Trimer10.17Only_Series3_44C/processed_series3_44C_all_images/TrimerOnly_Series-3_44C_1ms_042_Q.chi']
    
    outliers_1ms[1]
    > [5]
    
    outliers, outlier_index = svd_outliers(arr=on_array, flist=on_files, q=q, cutoff=2.5, save_dir='/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/ANALYSIS/test_dir/', 
                               save_name=str(t) + '_svd_outliers.csv')
                               
    outliers
    > ['/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Trimer10.17Only_Series3_44C/processed_series3_44C_all_images/TrimerOnly_Series-3_44C_1ms_042_Q.chi']
    
    outlier_index
    > [5]
    '''
    
    
    # Build matrix 
    matrix = np.matrix([i for i in arr]).transpose()
    
    # Run SVD
    u,s,v = svd(matrix, full_matrices=False)
    
    # Plot first left vector for laser on data (sanity check)
    plt.plot(q, -u[:, 0])
    plt.xlabel('Scattering Vector')
    plt.ylabel('Scattering Intensity (Log Scale)')
    plt.yscale('log')
    plt.title('Scattering Curve \nFirst Left Vector')
    
    make_dir(save_dir)
    
    plt.savefig(str(save_dir) + str(save_name) + '_left_vector1.png')
    plt.show()
    
    
    # Initialize lists for storing outliers
    outliers = []

    # Find outliers 
    for x in v[0]:
        # High outliers
        out_high = (np.where((x > v[0].mean()+v[0].std()*cutoff))[1])
        
        # Low outliers
        out_low = (np.where((x < v[0].mean()-v[0].std()*cutoff))[1])
    
    # Append outliers to list
    for o in out_high:
        outliers.append(o)
        
    for o in out_low:
        outliers.append(o)

    outliers = list(set(outliers))
        
    
    # Initialize list to store outlier file names
    outlier_files =[]

    # Get outlier files for scattering curves
    for o in outliers:
        outlier_files.append(flist[o])

    # Check/print outlier list 
    if len(outlier_files) > 0:
        print("\033[4mOutliers Determined by SVD at " + str(cutoff) + " STD: \033[0m" )
        for o in outlier_files:
            print(o)
    else:
        print('\033[1mNo Outliers Detected!\033[1m')

        
    # Check for and make directory to store output files
    make_dir(save_dir)
    
    # keep only unique outlier files in on_outlier_files
    outlier_files = list(set(outlier_files))
    
    # Write outlier files list to files
    np.savetxt(str(save_dir) + str(save_name) + '.csv', outlier_files, fmt="%s", delimiter=',')
            
            
    return outlier_files, outliers


def iterative_chi(arr, flist, chi_cutoff=1.5, outfile='outliers.csv', calls=1):
    '''
    Function to run chi-square analysis iteratively to find and discard all outliers
    above the indicated cutoff. Function iterates until no outliers are found. 
    
    Parameters:
    -----------
    arr : np.array
        Array containing scattering data to run iterative chi square test on. 
        
    flist : list
        Files associated with scattering data in input data array. 
        
    chi_cutoff (optional) : float
        Chi value to use as cutoff for determining outliers. All curves with a 
        chi value greater than the the chi_cutoff will be counted as outliers. 
        The default value is 1.5. 
        
    outfile (optional) : str
        File name, including full path, of where to store the outlier list. The 
        default value is a file 'outfiles.csv' in the current working directory. 

    calls (optional) : int
        Starting chi iteration number. Calls tracks the number of iterations run. 
        
        
    Returns:
    --------
    Does not return anything. List of chi outlier files will be saved to the indicated outfile. 
    
    Examples:
    ---------
    iterative_chi(arr=data_arr, flist=unique_files, chi_cutoff=2.5, outfile='/datacommons/dhvi-md/AshleyB/tmp/1ms_chi_outliers', calls=1)
    > Iteration: 1
      Number of outliers found: 5
      Number of laser on files left: 95
      /datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Trimer10.17Only_Series3_44C/processed_series3_44C_all_images/TrimerOnly_Series-3_44C_1ms_074_Q.chi
      /datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Trimer10.17Only_Series3_44C/processed_series3_44C_all_images/TrimerOnly_Series-3_44C_1ms_042_Q.chi
      /datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Trimer10.17Only_Series3_44C/processed_series3_44C_all_images/TrimerOnly_Series-3_44C_1ms_079_Q.chi
      /datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Trimer10.17Only_Series3_44C/processed_series3_44C_all_images/TrimerOnly_Series-3_44C_1ms_013_Q.chi
      /datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Trimer10.17Only_Series3_44C/processed_series3_44C_all_images/TrimerOnly_Series-3_44C_1ms_020_Q.chi
      Iteration: 2
      Number of outliers found: 4
      Number of laser on files left: 91
      /datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Trimer10.17Only_Series3_44C/processed_series3_44C_all_images/TrimerOnly_Series-3_44C_1ms_070_Q.chi
      /datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Trimer10.17Only_Series3_44C/processed_series3_44C_all_images/TrimerOnly_Series-3_44C_1ms_012_Q.chi
      /datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Trimer10.17Only_Series3_44C/processed_series3_44C_all_images/TrimerOnly_Series-3_44C_1ms_031_Q.chi
      /datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Trimer10.17Only_Series3_44C/processed_series3_44C_all_images/TrimerOnly_Series-3_44C_1ms_092_Q.chi
      Iteration: 3
      Number of outliers found: 1
      Number of laser on files left: 90
      /datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Trimer10.17Only_Series3_44C/processed_series3_44C_all_images/TrimerOnly_Series-3_44C_1ms_051_Q.chi
      Iteration: 4
      Number of outliers found: 0
      Number of laser on files left: 90
    '''
    
    # Calculate average curve
    avg_curve = arr.mean(axis=0)
    
    # Calculate standard deviation
    std = arr.std(axis=0)
    
    # Get n
    n = arr.shape[1]
    
    # Calculate chi sqaure value
    chi = np.sum( (arr - avg_curve)**2 / std**2 , axis=1) / (n-1)

    # Find outliers
    diff_outliers = np.where(chi >= chi_cutoff)
    print('Iteration: ' + str(calls))
    print('Number of outliers found: ' + str(len(diff_outliers[0])))
    
    # remove outliers from data array
    arr = np.delete(arr, diff_outliers, axis=0)
    print('Number of laser on files left: ' + str(len(arr)))
    
    # save outlier file list
    outlier_files = []
    
    for d in diff_outliers[0]:
        print(flist[d])
        outlier_files.append(flist[d])

    
    # Write outlier files list to files
    with open(outfile, "a") as a:
        for f in outlier_files:
            a.write(str(f) + "\n")
        
    # remove outliers from data file list
    for f in flist:
        if f in outlier_files:
            flist.remove(f)
            
    
    # Iterate
    if len(diff_outliers[0]) == 0:
        return 
        
    else:
        iterative_chi(arr, flist, chi_cutoff, outfile, calls+1)


def remove_outliers(flist, olist, fslice=None):
    '''
    Function to remove outlier files from a file list. Function
    takes a file list and an outlier list as input. The file list 
    can be made with the `make_flist` function and the outlier
    list can be made with the `svd_outliers` function or by loading
    a list of outliers from a saved file. The function will return 
    the flist with outlier files removed and a list of the outlier 
    files. Function also prints statements to indicate if the the
    number of files remaining after the outliers are removed matches
    the expected number of files based on the input flist and olist
    lengths (green = match, red = mismatch). 

    Parameters:
    -----------
    files : list
        Name of list storing file names including full path.
        
    olist : list
        File of the list storing outlier file names, including full path, 
        to be removed. 
        
    fslice (optional) : list
        List containing integers that will slice the replica 
        number from the file name. If set to None, then the 
        entire file name will be used. The default value is 
        None. Note that using the full file names could lead to
        failure to remove outliers from a list when trying to
        remove laser_off curves from laser_on or difference
        curve file sets. 
        
    Returns:
    --------
    cleaned : list
        List containing files with outliers removed. 
        
    olist : list
        List containg the outlier files
    '''
    
     # test if input parameters are proper data type
    #try:
    if not isinstance(flist, list):     
        raise TypeError('\033[1;91mTypeError: Oops! flist must be list type. Try again...\033[1;91m')
            
    if not isinstance(olist, list):     
        raise TypeError('\033[1;91mTypeError: Oops! olist must be list type. Try again...\033[1;91m')
        
    if not isinstance(fslice, (list, type(None))):     
        raise TypeError('\033[1;91mTypeError: Oops! fslice must be list type. Try again...\033[1;91m')

    '''
    except TypeError as e:
        print(e.args[0])
        if isinstance(e, TypeError):
            if e.args[0][18] == 'l':
                flist = input('TypeError: Enter a new list for flist parameter. Make sure files in flist include full path: ')
                
            elif e.args[0][17] == 'o':
                olist = input('TypeError: Enter a new list for olist parameter. Make sure files in olist include full path: ')
              
            elif e.args[0][18] == 's':
                fslice = input('TypeError: Enter a new list value for fslice. If unsure of proper fslice use None instead: ')
    '''        
        
                
    
    # get length of file list before outliers are removed
    print('\033[92mNumber of files loaded: \033[92m' + str(len(flist)))
    
    print('\033[91mNumber of outliers to remove: \033[91m' + str(len(olist)))
    
    # make copy of flist
    cleaned = flist.copy()
    
    # get file slice for replica number
    if fslice is not None:
        fs = fslice
    
    # loop over all loaded files
    for f in flist:
        
        # check that there are outliers
        if len(olist) != 0:
        
            # get file slice
            if fslice is None:
                fs = [0, len(f)]

            # loop over outlier list    
            for o in olist:
                # remove outliers
                if f[fs[0]:fs[1]] in o[fs[0]:fs[1]]:
                    print(str(f) + '\033[91m is an outlier\033[91m')
                    if o[fs[0]:fs[1]] in f[fs[0]:fs[1]]:
                        cleaned.remove(f)
                    else:
                        print('\033[1;92mOutlier file not removed because outlier file is not in file list. Ignore warnings about mismatch between number of outlier files remaining \033[1;92m')
    
    if len(cleaned) == len(flist) - len(olist):
        print('\033[1;92mNumber of files remaining after outliers removed: \033[1;92m' + str(len(cleaned)))
        
    
    
    else:
        print('\033[1;91mWARNING! Number of files remaining after outliers removed: ' + str(len(cleaned)) + ' does not match the number expected based on the size of input file and outlier lists\033[1;91m')
           
    return cleaned, olist 

def saxs_scale(ref, scale, dataset, err=False, delim=',', mask=0, qmin=1.5, qmax=2.6, outfile=None, outdir=None):
    '''
    Function to scale a curve to a reference curve. Scaling is based on 
    and algebraic method using the equation:
            scalar = np.dot(ref, ref) / np.dot(ref, buffer)
    Saves scaled curves and plots. Returns an array of scaled curve in 
    the group. 
    
    Parameters:
        
    ref : str
        File name, including full path, to the files containing curve
        to used as reference for scaling the buffer curve. Assumes that
        the file has three columns representing scattering vector, 
        scattering intensity, and standard error, respectively. 

    scale : str
        File name, including full path, to file containing the buffer curve
        to scale. Assumes that the file has three columns representing the 
        scattering vector, scattering intensity, and standard error, 
        respectively. 
        
    dataset : str
        String to describe the dataset that is being processed. Will be used
        in plot titles and file names. 
        
    err (optional) : bool
        Boolean indicating if the file containing the curves also contain errors 
        for the curves. If set to True, then the errors will be propagated. The
        default value is False. 
        
    delim (optional) : str
        Delimitter used in data files. Default value is comma. 
        
    mask (optional) : int
        Number of rows to skip when importing curves. When set to 0 all rows are imported.
        Default value is 0. 
        
    qmin (optional) : float
        Minimum scattering vector value for curve scaling. Default value is 1.5 Å^-1. 
        
    qmax (optional) : float
        Maxmimum Q value for curve scaling. Default value is 2.6 Å^-1. 
        
    outfile (optional) : str
        Name to use for output scaled curve file. When set to None, no output curves will be saved. Default
        value is None.

    outdir (optional) : str
        Path to directory where output files will be saved. Default value is current
        working directory. 
        
    Returns:
    --------
    scaled : np.array
        Numpy array containing the full scaled curved that was saved to file.
        
    Raises:
    -------  
    IndexError
        When errors are indicated but do not exist in the given file
        
    Examples:
    ----------
    scaled = saxs_scale(ref='/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Trimer10.17Only_Series3_44C/processed_series3_44C_all_images/TrimerOnly_Series-3_44C_1ms_017_Q.chi',
                        scale='/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Trimer10.17Only_Series3_44C/processed_series3_44C_all_images//TrimerOnly_Series-3_44C_-5us_017_Q.chi', 
                        dataset='test', err=False, delim=' ', mask=12, qmin=1.4, qmax=1.6, outfile='scaled.csv', outdir='/datacommons/dhvi-md/AshleyB/tmp/')
    
    scaled
    > (array([7.89046914, 7.67275974, 7.53970047, ..., 0.96371704, 0.96319655,
        0.96333235]),
       array([0., 0., 0., ..., 0., 0., 0.]))
       
    scaled, errs = saxs_scale(ref='/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Trimer10.17Only_Series3_44C/processed_series3_44C_all_images/TrimerOnly_Series-3_44C_1ms_017_Q.chi',
                              scale='/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Trimer10.17Only_Series3_44C/processed_series3_44C_all_images//TrimerOnly_Series-3_44C_-5us_017_Q.chi', 
                              dataset='test', err=False, delim=' ', mask=12, qmin=1.4, qmax=1.6, outfile='scaled.csv', outdir='/datacommons/dhvi-md/AshleyB/tmp/')
    
    scaled
    > array([7.89046914, 7.67275974, 7.53970047, ..., 0.96371704, 0.96319655,
        0.96333235])
        
    errs
    > array([0., 0., 0., ..., 0., 0., 0.]))
       
    
       
        
    ** Note that the returned scaled curve DOES NOT contain scattering vector values. The scattering
       vectors are unchanged by the scaling so you can import them from the raw data file. **
    '''
    
    try:
        
        # load data
        print('Loading data...')
        prot = load_saxs(file=ref, delim=delim, mask=mask)
        buf = load_saxs(file=scale, delim=delim, mask=mask)        
        
                
    except IndexError as e:
        tb = e.__traceback__
        print('\033[91m' + str(e.args[0]) + '\033[91m') 
        traceback.print_tb(tb)

        print('\033[93mIndexError: Attempted to propagate errors but no errors were loaded. Changing err to False...]')
        err = False
            

    finally:
        # load data
        print('Loading data...')

        prot = load_saxs(file=ref, delim=delim, mask=mask)
        buf = load_saxs(file=scale, delim=delim, mask=mask)

        
        data = []        

        # plot data prior to scaling
        ax = plt.axes([0.125,0.125, 5, 5])
        plt.plot(prot[:,0], prot[:,1], label='reference')
        plt.plot(buf[:,0], buf[:,1], label='pre-scale')
        plt.legend(loc='best', fontsize=60)
        plt.xlabel('Scattering Vector (Å' + r'$^{-1}$' + ')', fontsize=60)
        plt.ylabel('Scattering Intensity', fontsize=60)
        plt.title(str(dataset) + ' Before Scaling', fontsize=70)
        plt.xticks(fontsize=55)
        plt.yticks(fontsize=55)
        plt.set_cmap('viridis')
    
        a = plt.axes([-5, 0.5, 4, 4])
        plt.plot(prot[:,0], prot[:,1], label='protein')
        plt.plot(buf[:,0], buf[:,1], label='buffer')
        plt.legend(loc='best', fontsize=60)
        plt.xlabel('Scattering Vector (Å' + r'$^{-1}$' + ')', fontsize=60)
        plt.ylabel('Scattering Intensity', fontsize=60)
        plt.title(str(dataset) + ' Before Scaling', fontsize=70)
        plt.xticks(fontsize=55)
        plt.yticks(fontsize=55)
        plt.set_cmap('viridis')
        plt.xlim([qmin, qmax])
    
        # format borders
        for axis in ['top','bottom','left','right']:
            a.spines[axis].set_linewidth(5)

        # mark inset
        mark_inset(ax, a, loc1=1, loc2=4, fc="none", ec="0.5", 
                   linewidth=4)
    
        # make directory for storing saved plots
        make_dir(f=outdir+'/PLOTS/')
    
        #save plot
        plt.savefig(outdir + '/PLOTS/' + str(dataset) + '_before_scale.png', 
                bbox_inches='tight')
        # show plot 
        plt.show()
    
        # mask data
        prot_mask = prot[prot[:,0] > qmin]
        prot_mask = prot_mask[prot_mask[:,0] < qmax]
        buf_mask = buf[buf[:,0] > qmin]
        buf_mask = buf_mask[buf_mask[:,0] < qmax]
    
        # calculate scalar using algebraic method
        s = np.dot(prot_mask[:,1], buf_mask[:, 1])
        unitary = np.dot(prot_mask[:,1], prot_mask[:,1])
        scalar = unitary / s
        

    
        # scale buffer curve
        scaled = buf[:,1] * scalar
    
        # create array for propagated  errors
        if err is True:
            buf_err = buf[:,2] * scalar
        else:
            buf_err = np.zeros(shape=(scaled.shape))

        # plot data prior to scaling
        ax = plt.axes([0.125,0.125, 5, 5])
        plt.plot(prot[:,0], prot[:,1], label='reference')
        plt.plot(buf[:,0], buf[:,1], label='pre-scaled')
        plt.plot(buf[:,0], scaled, label='scaled', alpha=0.5)
        plt.legend(loc='best', fontsize=60)
        plt.xlabel('Scattering Vector ' + r'$^{-1}$' + ')', fontsize=60)
        plt.ylabel('Scattering Intensity', fontsize=60)
        plt.title(str(dataset) + ' After Scaling', fontsize=70)
        plt.xticks(fontsize=55)
        plt.yticks(fontsize=55)
        plt.set_cmap('viridis')
    
        a = plt.axes([-5, 0.5, 4, 4])
        plt.plot(prot[:,0], prot[:,1], label='reference')
        plt.plot(buf[:,0], buf[:,1], label='pre-scaled')
        plt.plot(buf[:,0], scaled, label='scaled', alpha=0.5)
        plt.legend(loc='best', fontsize=60)
        plt.xlabel('Scattering Vector ' + r'$^{-1}$' + ')', fontsize=60)
        plt.ylabel('Scattering Intensity', fontsize=60)
        plt.title(str(dataset) + ' After Scaling', fontsize=70)
        plt.xticks(fontsize=55)
        plt.yticks(fontsize=55)
        plt.set_cmap('viridis')
        plt.xlim([qmin, qmax])

        for axis in ['top','bottom','left','right']:
            a.spines[axis].set_linewidth(5)

        # mark inset
        mark_inset(ax, a, loc1=1, loc2=4, fc="none", ec="0.5", 
                   linewidth=4)
        
        if outfile is not None:
            # save scaled curve
            np.savetxt(str(outdir) + '/' + str(outfile), np.c_[buf[:, 0], scaled, buf_err], delimiter=",")
        
            #save plot
            plt.savefig(str(outdir) + '/PLOTS/' + str(dataset) + '_after_scale.png', 
                        bbox_inches='tight')
    
        # show figure
        plt.show()
    
    return scaled, buf_err


def saxs_sub(ref, data, delim_ref=',', delim_data=',', err=False, ref_skip=1, data_skip=1,
             outfile=None, outdir='./'):
    '''
    A function to correct SAXS scattering curve by subtracting a reference scattering curve.
    Input reference and data are assumed to be a SAXS scattering curve output from bootstrap
    averaging
    in a text file with 3 columns for Q, I, and STD_I. Saves subtracted curves with
    designated file names in the designated folder. 
    
    Parameters:
    -----------
    ref : str
        File name, including full path, to the file containing the reference SAXS scattering
        curve. File is assumed to be a plain text file with format from the output of bootstrap
        averaging and containing three columns, Q, I, and STD_I. 
        
    data : str
        File name, including full path, of file containing the SAXS scattering curve that
        will be corrected. File is assumed to be a plain text file with two columns, q and
        i. 
        
    delim_ref (optional) : str
        Delimitter used in file storing reference scattering curve. Default is ' '. 
    
    delim_data (optional) : str
        Delimitter used in fle storing scattering curve to be corrected. Default is ' '. 
        
    err (optional) : bool
        Boolean indicating if error should be propagated. Errors will be propagated 
       when set to True. Default value is False. 
        
    ref_skip (optional) : str
        Rows to skip in reference file. If set to None then no rows will be skipped. 
        Default value is None. 
                
    data_skip (optional) : str
        Rows to skip in data file. If set to None then no rows will be skipped. 
        Default value is None. 
        
    outfile (optional) : str
        File name to save subtracted curve to. If set to None, then no file will be saved. 
        Default value is None. 
        
    outdir (optional) : str
        Name of directory to save outfile in. If the directory does not exist, then it will
        be made. Default value is the current working directory.
        
    Returns:
    --------
    corrected : DataFrame
        DataFrame containing the subtracted curve. 
        
    Examples:
    ---------
    ref, data, correction = saxs_sub(ref='/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Trimer10.17Only_Series3_44C/processed_series3_44C_all_images/TrimerOnly_Series-3_44C_1ms_017_Q.chi',
                                data='/datacommons/dhvi-md/AshleyB/tmp/scaled.csv', delim_ref=' ', delim_data=',', err=False, ref_skip=12, data_skip=0, outfile=None, 
                                 outdir='/datacommons/dhvi-md/AshleyB/tmp/')
                                 
    ref
    > array([[0.01947379, 7.90474967],
             [0.02091628, 7.67862152],
             [0.02235877, 7.52825252],
             ...,
             [2.56431318, 0.96828936],
             [2.56546711, 0.96746814],
             [2.56662079, 0.96760997]])
             
    data
    > array([[0.01947379, 7.89046914, 0.        ],
             [0.02091628, 7.67275974, 0.        ],
             [0.02235877, 7.53970047, 0.        ],
             ...,
             [2.56431318, 0.96371704, 0.        ],
             [2.56546711, 0.96319655, 0.        ],
             [2.56662079, 0.96333235, 0.        ]])
             
    correction[:5]
    > [-0.014280532015460423,
       -0.005861783138817245,
       0.0114479564882366,
       0.014281655270111493,
       0.014171961779569031]
    '''
    
    # load reference data
    print('Loading reference file...')
    ref = np.loadtxt(fname=ref, delimiter=delim_ref, skiprows=ref_skip)
    
    # load data to be corrected
    print('Loading data file...')
    data = np.loadtxt(fname=data, delimiter=delim_data, skiprows=data_skip)
    
    # set up dataframe to save corrected data
    #corrected = pd.DataFrame(data.iloc[:,0])
    correted = []
    
    # set empty list for data correction
    correction = []

    # loop over all i values in data and ref
    print('Subtracting reference from data...')
    for i,r in zip(data, ref):
        # calculate correction 
        c = i[1] - r[1] 
        
        # append list of corrected i values to corrected dataframe 
        #    corrected.append([data[:,0], correction])
    
        #loop over all i_dts values in data and ref
        #print('Calculating Error...')
        #for i, r in zip(data.iloc[:,2], ref.iloc[:, 2]):

        # append corrected and error to correction
        correction.append(c)
        
        # calculate the error
        if err is True:
            error = []
            e = np.sqrt(i[1]**2 + r[1]**2)
            error.append(e)
            return error
    
    # append error to list of  subtracted values
    #corrected.append(loc=2, column='i_std', value=error)
    
    if outfile is not None:
        if err is True:
            print('Saving corrected data...')
            make_dir(outdir)
            np.savetxt(fname=str(outdir + outfile), X=np.c_[data[:,0], correction, error], 
                       header='q,scaled_i,scaled_err', 
                       delimiter=',')
        if err is False:
            print('Saving corrected data...')
            make_dir(outdir)
            np.savetxt(fname=str(outdir + outfile), X=np.c_[data[:,0], correction], 
                       header='q,scaled_i', 
                       delimiter=',')
    
    return ref, data, correction


def auc_outlier(time_delay, file, upper=0.75, lower=0.25, delim=',',f_col='file', 
                auc_col='auc_simpsons', outdir='./'):
    '''
    Function to find outliers for SAXS difference curves based on
    area under the curve calculations. Determines the outliers by 
    finding images with AUC that are outside the interquartile range 
    for the input time delay AUC. Boxplot of data is also produced. 
    Detected outliers will be saved to CSV file, as will the data 
    with the outliers removed. 
    
    Parameters:
    ------------
    time_delay : str
        Time delay for input data set. 
        
    file : str
        File containing the area under
        the curve calculations for each
        file in a SAXS time delay data 
        set. The input data should be a 
        CSV file with the file name and 
        AUC calculation. 
        
    upper (optional) : float
        Value to define upper quantile. 
        Default is 0.75.
        
    lower (optional) : float
        Valur to define the lower quantile.
        Default is 0.25.
        
    delim (optional) : str
        Delimiter used in input CSV file. 
        Default value is ','. 
        
    f_col (optional) : str
        Name of column containing file name 
        in input CSV file. Default column 
        name is 'file'. 
        
    auc_col (optional) : str
        Name of column containing the AUC 
        calculation in inut CSV file. Default 
        column name is 'auc_simpsons'. 
        
    outdir (optional) : str
        Name of directory to store output files 
        contraining detected outliers including 
        full path. Default is current working directory. 
        The default file name will be <time_delay.csv>.
        If directory doesn't exist, it will be made. 

        
    Returns:
    ---------
    Pandas DataFrame containing file number,
    file name, and the AUC calculation. 
    
    Pandas DataFrame containing the file name
    and AUC calculation for the difference 
    curves determined to be outliers. 
    
    Pandas DataFrame which is the input DataFrame
    with the determined outliers removed. 
    
    '''
    # Load auc data as a data frame
    df = pd.read_csv(file, header=0, delimiter=delim)
    
    # Get IQR
    q1 = df[auc_col].quantile(lower)
    q3 = df[auc_col].quantile(upper)
    iqr = q3 - q1
    
    # Set outlier detection threshold
    outlier_top_lim = q3 + 1.5 * (q3 - q1)
    outlier_bottom_lim = q1 - 1.5 * (q3 - q1)
    
    # find outliers
    outliers = []
    dat = []
    for file, auc in zip(df[f_col], df[auc_col]):
        if (auc > outlier_top_lim) or (auc < outlier_bottom_lim):
            #print(file, auc)
            outliers.append([file, auc])
        else:
            dat.append([file, auc]) 
    print(len(outliers))
            
    # save outliers to file
    make_dir(outdir)
    outlier_outfile = str(outdir + time_delay) + '_outliers.csv'
    outliers = pd.DataFrame(outliers, columns=['file', 'auc'])
    outliers.to_csv(outlier_outfile)
    
    # save filtered data to file
    make_dir(outdir)
    clean_outfile = str(outdir + time_delay) + '_cleaned.csv'
    clean = pd.DataFrame(dat, columns=['file', 'auc'])
    clean.to_csv(clean_outfile)
    
    # plot data
    sns.boxplot(y=df[auc_col])
    for file, auc in zip(df[f_col], df[auc_col]):
        if (auc > outlier_top_lim) or (auc < outlier_bottom_lim):
            plt.text(.5, auc, f' {file}', ha='left', va='center')
    plt.xlabel(str(time_delay))
    plt.tight_layout()
    plt.show()
    
    return (df, outliers, dat)


def move_outliers(flist, fpath='./', col_name='file', quar_dir='./quar/'):
    '''
    Function to move a list of files to a new directory. File names are not changed. Files 
    moved to designated directory. File list is assumed to be contained within a column
    of a pandas DataFrame. 
    
    Parameters:
    -----------
    flist : DataFrame
        DataFrame in which one column contains the list of files to be moved.
        
    fpath (optional) : str
        Path to directory where loaded files are currently stored. Defualt is current 
        working directory. 
        
    col_name (optional) : str
        Name of DataFrame column containing file list. Default is 'file'.
        
    quar_dir (optional) : str
        Name of directory, including full path, to move files to. If the directory does 
        not exist, it will be made. Default is a new directory 'quar' in the current
        working directory. 
        
    Returns:
    --------
    
    '''
    make_dir(quar_dir)
    
    # loop over files in list and move them to new directory
    for f in flist[col_name]:
        shutil.move(str(fpath + f), 
                    str(quar_dir))
    return