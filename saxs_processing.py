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

def saxs_scale(ref, scale, dataset, err=False, delim=',', mask=0, qmin=1.5, qmax=2.6, outdir='./'):
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
        in plot titles and output file names. 
        
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

    outdir (optional) : str
        Path to directory where scaled curve will be saved. Default value is current
        working directory. 
        
    Returns:
    --------
    scaled : np.array
        Numpy array containing the full scaled curved that was saved to file.
        
    Raises:
    -------
    TypeError 
        When the reference, or scale file or delim is not str type, mask is not int type, or err is not bool type. 
        
    ValueError
        When the given delim does not match the delimitter in the given file
        
    IndexError
        When errors are indicated but do not exist in the given file
        
    Examples:
    ----------
    scaled = saxs_scale(ref='/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Complex10.17-I5_TSeries-11_44C/processed/Complex10.17-I5_TSeries-11_44C_100ms_001_Q.chi',
                        scale='/datacommons/dhvi-md/TR_T-jump_SAXS_Mar2023/Complex10.17-I5_TSeries-11_44C/processed/Complex10.17-I5_TSeries-11_44C_-5us_001_Q.chi', 
                        dataset='test', err=False, delim=' ', mask=12, qmin=1.4, qmax=1.6, outdir='./tmp/')
    
    scaled
    > array([8.59403483, 8.28879774, 7.95483028, ..., 0.95764088, 0.95691427,
       0.95716287])
       
        
    ** Note that the returned scaled curve DOES NOT contain scattering vector values. The scattering
       vectors are unchanged by the scaling so you can import them from the raw data file. **
    '''
    
    try:
        if not isinstance(ref, str):
            raise TypeError('TypeError: ref file must be str type. Try again...')
            
        if not isinstance(scale, str):
            raise TypeError('TypeError: scale file must be str type. Try again...')
        
        if not isinstance(dataset, str):
            raise TypeError('TypeError: dataset must be str type. Try again...')
            
        if not isinstance(err, bool):
            raise TypeError('TypeError: err must be bool type. Try again...')
            
        if not isinstance(delim, str):
            raise TypeError('TypeError: delim must be str type. Try again...')
            
        if not isinstance(mask, int):
            raise TypeError('TypeError: mask must be int type. Try again...')
            
        if not isinstance(qmin, float):
            raise TypeError('TypeError: qmin must be int type. Try again...')
        
        if not isinstance(qmax, float):
            raise TypeError('TypeError: qmax must be int type. Try again...')
            
        if not isinstance(outdir, str):
            raise TypeError('TypeError: outdir must be str type. Try again...')
            

        # load data
        print('Loading data...')
        prot = load_saxs(file=ref, delim=delim, mask=mask)
        buf = load_saxs(file=scale, delim=delim, mask=mask)        
        
                
    except Exception as e:
        tb = e.__traceback__
        print('\033[91m' + str(e.args[0]) + '\033[91m') 
        traceback.print_tb(tb)
        if isinstance(e, TypeError):
            if e.args[0][11] == 'r':
                print('\033[93mTypeError: Enter a new str value for ref file name including full path (do not include quotations): \033[93m')
                ref = input()
                
            elif e.args[0][11] == 's':
                print('\033[93mTypeError: Enter a new str value for scale file name including full path (do not include quotations): \033[93m')
                scale = input()
              
            elif e.args[0][11:12] == 'da':
                print('\033[93mTypeError: Enter a new str value for dataset: \033[93m')
                dataset = input()
                
            elif e.args[0][11] == 'e':
                print('\033[93mTypeError: Enter a new bool value for err: \033[93m')
                err = input()
                
            elif e.args[0][11:12] == 'de':
                print('\033[93mTypeError: Enter a new str value for delim: \033[93m')
                delim = input()
                
            elif e.args[0][11] == 'm':
                print('\033[93mTypeError: Enter a new int value for mask: \033[93m')
                mask = int(input())
                
            elif e.args[0][12:15] == 'min':
                print('\033[93mTypeError: Enter a new int value for qmin: \033[93m')
                qmin = float(input())
                
            elif e.args[0][12:15] == 'max':
                print('\033[93mTypeError: Enter a new int value for qmax: \033[93m')
                qmax = float(input())
                
            elif e.args[0][11] == 'o':
                print('\033[93mTypeError: Enter a new str value for outdir path: \033[93m')
                dataset = input()
                
        elif isinstance(e, ValueError):
            print('\033[93mValueError: File has a different delim. Enter a new str value for delim (do not include quotations): \033[93m')
            delim = str(input())
        
        elif isinstance(e, IndexError):
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
        plt.xlabel('Scattering Vector ' + r'$^{-1}$' + ')', fontsize=60)
        plt.ylabel('Scattering Intensity', fontsize=60)
        plt.title(str(dataset) + ' Before Scaling', fontsize=70)
        plt.xticks(fontsize=55)
        plt.yticks(fontsize=55)
        plt.set_cmap('viridis')
    
        a = plt.axes([-5, 0.5, 4, 4])
        plt.plot(prot[:,0], prot[:,1], label='protein')
        plt.plot(buf[:,0], buf[:,1], label='buffer')
        plt.legend(loc='best', fontsize=60)
        plt.xlabel('Scattering Vector ' + r'$^{-1}$' + ')', fontsize=60)
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

    
        # save scaled curve
        np.savetxt(str(outdir) + '/' + str(dataset) + '_buff_scale.csv', 
                   np.c_[buf[:, 0], scaled, buf_err], delimiter=",")

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

        #save plot
        plt.savefig(outdir + '/PLOTS/' + str(dataset) + '_after_scale.png', 
                    bbox_inches='tight')
    
        # show figure
        plt.show()
    
    return scaled

def saxs_sub(ref, data, delim_ref=',', delim_data=',', err=False, ref_skip=1, data_skip=1,
             outfile=None, outdir=','):
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