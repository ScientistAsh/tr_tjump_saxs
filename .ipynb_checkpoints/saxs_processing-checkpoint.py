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
        
    Raises:
    --------
    TypeError:
        When arr is not np.ndarray, flist is not list, cutoff is not float, or save_dir or save_name is 
        is not str data types. Exception handling will prompt you to enter a new value for the incorrect
        parameter. 
        
    Examples:
    ----------
    on_outliers = svd_outliers(arr=on_array, flist=on_files, q=q, cutoff=2.5, save_dir='/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/ANALYSIS/test_dir/', 
                               save_name=str(t) + '_svd_outliers.csv')
    on_outliers
    > (['/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Trimer10.17Only_Series3_44C/processed_series3_44C_all_images/TrimerOnly_Series-3_44C_5us_030_Q.chi'],
     [29])
     
    on_outliers[0]
    > ['/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Trimer10.17Only_Series3_44C/processed_series3_44C_all_images/TrimerOnly_Series-3_44C_5us_030_Q.chi']
    
    on_outliers[1]
    > [29]
    
    outliers, outlier_index = svd_outliers(arr=on_array, flist=on_files, q=q, cutoff=2.5, save_dir='/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/ANALYSIS/test_dir/', 
                               save_name=str(t) + '_svd_outliers.csv')
    '''
    
    # test if input parameters are proper data type
    try:
        if not isinstance(arr, np.ndarray):     
            raise TypeError('TypeError: Oops! arr must be np.ndarray type. Try again...')
            
        if not isinstance(flist, list):     
            raise TypeError('TypeError: Oops! flist must be list type. Try again...')
        
        if not isinstance(cutoff, float):     
            raise TypeError('TypeError: Oops! cutoff must be float type. Try again...')
        
        if not isinstance(save_dir, str):     
            raise TypeError('TypeError: Oops! save_dir must be str type. Try again...')
            
        if not isinstance(save_name, str):     
            raise TypeError('TypeError: Oops! save_name must be str type. Try again...')
            
    except TypeError as e:
        print(e.args[0])
        if isinstance(e, TypeError):
            if e.args[0][17] == 'a':
                flist = input('TypeError: Enter a new np.ndarray for arr parameter: ')
                
            elif e.args[0][17] == 'f':
                delim = input('TypeError: Enter a new list for flist parameter. Make sure files in flist include full path: ')
              
            elif e.args[0][11] == 'c':
                mask = int(input('TypeError: Enter a new float value for cutoff: '))  
                
            elif e.args[0][22] == 'd':
                err = input('TypeError: Enter a new str value for save_dir (remember to include "/" after directory name): ')
            
            elif e.args[0][22] == 'n':
                err = input('TypeError: Enter a new str value for save_name: ')

    
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
                    cleaned.remove(f)
    
    if len(cleaned) == len(flist) - len(olist):
        print('\033[1;92mNumber of files remaining after outliers removed: \033[1;92m' + str(len(cleaned)))
    
    else:
        print('\033[1;91mWARNING! Number of files remaining after outliers removed: ' + str(len(cleaned)) + ' does not match the number expected based on the size of input file and outlier lists\033[1;91m')
           
    return cleaned, olist 