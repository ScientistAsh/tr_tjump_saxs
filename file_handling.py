'''
tr_tjump_saxs: file_handle.py
Date created (mm-dd-yy): 04-01-2023

This module is part of the tr_tjump_saxs package for processing and anlayzing pump-probe time resolved, 
temperature-jump small angle X-ray scattering data sets. This module includes functions for creating new 
output directories, loading saxs curves, and plotting saxs curves.

Ashley L. Bennett, PhD
@ScientistAsh
'''

# import dependent modules
import numpy as np
import os 
import shutil
import warnings
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
import traceback 
import sys
import matplotlib.pylab as pl 
from time import sleep
from tqdm.notebook import tqdm



def make_dir(f):
    '''
    Function to assess if a directory exists. If directory does not exists, it will be made.
    If prompted to enter a new directory path, do use without using quotation marks to inclose
    the file name.
    
    Parameters:
    -----------
    f : str 
        Directory name including full path
        
    Returns:
    --------
    
    
    Raises:
    -------
    TypeError
        When f is not a str type
        
    Examples:
    ---------
    make_dir(f='./ANALYSIS/')
    
    
    
    '''
    try:
        if not isinstance(f, str):
            raise TypeError('Oops! f must be str type.  Try again...')
        
        elif not os.path.exists(f):
            os.makedirs(f)
        else:
            print('\033[1;92mDirectory already exists!\033[1;92m')
            
    except TypeError as e:
        tb = e.__traceback__
        print('\033[91mA TypeError occurred\033[91m')
        traceback.print_tb(tb)
        print('\033[93mEnter a new directory path string (do not include quotations): \033[93m')
        make_dir(input())
        
    return

def make_flist(directory='./', prefix=None, suffix=None):
    '''
    Function to create a list of files stored in a specific directory. 
    
    Parameters:
    -----------
    directory (optional) : str
        Directory with files to add to list. Returns a file list. 
        
    prefix (optional) : str
        Beginning of file names. Can be used to limit files appended if not all files in the
        directory need to be added to the list. When set to None, prefix parameter will not
        be used to filter the files appended to the file list. Default is None. 
        
    suffix (optional) : str
        Ending of file names. Can be used to limit files appended if not all files in the 
        directory need to be added to the file list. When set to None, prefix parameter will
        not be used to filter the files appended to the file list. Default is None.
        
    Returns:
    --------
    files : list containing the appended files. 
    
    Raises:
    -------
    TypeError
        When the given directory, prefix, or suffix is not str type.
        
    FileNotFoundError
        When the given directory does not exist
    '''
    
    try:
        if not os.path.exists(directory):
            raise FileNotFoundError('FileNotFoundError: Directory does not exist. Try again...')
            
        if not isinstance(directory, str):
            raise TypeError('TypeError: Oops! directory must be str type. Try again...')
           
        if prefix is not None and not isinstance(prefix, str):
            raise TypeError('TypeError: Oops! prefix must be str type. Try again...')
            
        if suffix is not None and not isinstance(suffix, str):
            raise TypeError('TypeError: Oops! suffix must be str type. Try again...')
    
    except (TypeError, FileNotFoundError) as e:
        tb = e.__traceback__
        print('\033[91m' +str(e.args[0]) +'\033[91m')
        traceback.print_tb(tb)
        
        if isinstance(e, TypeError):
            if e.args[0][17] == 'd':
                print('\033[93mEnter a new value for directory (do not include quotations): \033[93m')
                directory = input()
                
            elif e.args[0][17] == 'p':
                print('\033[93mEnter a new value for prefix (do not include quotations): \033[93m')
                prefix = input()
                
            elif e.args[0][17] == 's':
                print('\033[93mEnter a new value for suffix (do not include quotations): \033[93m')
                suffix = input()
        elif isinstance(e, FileNotFoundError):
            print('\033[93mEnter a new value for directory (do not include quotations): \033[93m')
            directory = input()
            
        make_flist(directory, prefix, suffix)
        
        #return e.args
        

    # create empty list for files
    files = []

    # get all files 
    if prefix is None and suffix is None:
        for f in tqdm(os.listdir(directory), desc='Getting files'):
            files.append(str(directory + f))
            
    # filter files by suffix
    elif prefix is None:
        for f in tqdm(os.listdir(directory), desc='Getting files'):
            if f.endswith(suffix):
                files.append(str(directory + f))
   
    # filter files by prefix
    elif suffix is None:
        for f in tqdm(os.listdir(directory), desc='Getting files'):
            if f.startswith(prefix):
                files.append(str(directory + f))

    # filter files by both prefix and suffix
    elif prefix is not None and suffix is not None:
        for f in tqdm(os.listdir(directory), desc='Getting files'):
            if f.startswith(prefix) & f.endswith(suffix):
                files.append(str(directory + f))
    
    print('Done loading ' + str(len(files)) + ' files!')
    
    # check that files are loaded into list
    if len(files) == 0:
        print('\033[1;93mWARNING: No files loaded\033[1;93m')
                        
    return files





def load_saxs(file, delim=' ', mask=0):
    '''
    Function to load a single SAXS difference curve as an array. Data file must be a flat text file
    delimited by space, tab, or comma. Automatically loads all columns in the given input file. To
    continue to use the difference curve data for further analysis call the function inside a 
    variable definition. The first dimension will slice rows and the second will slice columns. 
    
    Parameters:
    -----------
    
    file : str
        File including full path containing SAXS difference curve. File should be a simple space, comma, or 
        tab delimited text file.
    
    delim (optional) : str
        Type of delimiter used in data file. Accepted values are space (' '), tab ('\t'), and 
        comma (','). Default value is space (' '). 
        
    mask (optional) : int
        Number of rows to skip when loading files. Default values is 0. Useful for
        skipping rows with NaN or masked values. 

        
    Returns:
    --------
    data : np.array
        numpy array with a shape determined by input data. 
        
        
    Raises:
    -------
    TypeError 
        When the file or delim is not str type, mask is not int type, or err is not bool type. 
        
    ValueError
        When the given delim does not match the delimitter in the given file
        
    IndexError
        When errors are indicated but do not exist in the given file
        
    Examples:
    ----------
    curve = load_saxs(files[0], delim=' ', mask=12)
    
    curve
    > array([[ 0.01947379, -0.03160436],
            [ 0.02091628, -0.02051868],
            [ 0.02235877, -0.01261062],
            ...,
            [ 2.56431318,  0.00460323],
            [ 2.56546711,  0.00433085],
            [ 2.56662079,  0.00426167]])
    curve[0]
    > [ 0.01947379, -0.03160436]
    
    curve[:, 0]
    > array([0.01947379, 0.02091628, 0.02235877, ..., 2.56431318, 2.56546711,
       2.56662079])
       
    **(Note that when loading a typical SAXS scattering or difference curve where
       the first column is scattering vector this slicing yields the scattering vector)
       
    curve[:, 1]
    > array([-0.03160436, -0.02051868, -0.01261062, ...,  0.00460323,
        0.00433085,  0.00426167]
        
    **(Note that when loading a typical SAXS scattering or difference curve where
       the second column is scattering intensity (difference) this slicing yields the scattering intensity)
    '''
    
    try:
        if not isinstance(file, str):
            raise TypeError('\033[1;91mTypeError: file must be str type. Try again...\033[1;91m')
            
        if not isinstance(delim, str):
            raise TypeError('\033[1;91mTypeError: delim must be str type. Try again...\033[1;91m')
        
        if not isinstance(mask, int):
            raise TypeError('\033[1;91mTypeError: mask must be int type. Try again...\033[1;91m')

                
                
        data = np.loadtxt(file, delimiter=delim, skiprows=mask)
                
                
                
    except Exception as e:
        print(e.args[0])
        if isinstance(e, TypeError):
            if e.args[0][11] == 'f':
                file = input('TypeError: Enter a new str value for file name including full path (do not include quotations): ')
                
            elif e.args[0][11] == 'd':
                delim = input('TypeError: Enter a new str value for delim (do not include quotations): ')
              
            elif e.args[0][11] == 'm':
                mask = int(input('TypeError: Enter a new int value for mask: '))  
                
        elif isinstance(e, ValueError):
            delim = str(input('ValueError: File has a different delim. Enter a new str value for delim (do not include quotations): '))
        
    data = np.loadtxt(file, delimiter=delim, skiprows=mask)
                
    return data


def load_set(flist,  delim=' ', mask=0, err=False):
    '''
    Function to load a set of SAXS scattering curves with a specific file prefix/suffix 
    in a given directory. Input data has columns for q and I.
    
    Parameters:
    -----------
    flist : list
        List contining files, with full path, containing SAXS scattering or
        difference curves curves to run SVD on. Assumes that the input files 
        contain two columns with the scattering vector (q) and the 
        scattering intensity (i). If err=True, the the experimental error will
        also be loaded as long as the input data file contains a 3 column with 
        the experimental error. 
        
    delim (optional) : str
        Delimitter used in the input data files. Default value is a space (' '). 
        
    mask (optional) : int
        Number of rows to skip when loading files. Default values is 0. Useful for
        skipping rows with NaN or masked values. 
        
    err (optional) : bool
        Indicates if there is the column containing experimental error. If set 
        to False, then no errors will be loaded. Default value is False. 
        
        
    Returns:
    --------
    data : list 
        List containing the scattering intensity (i) vector for the loaded curves.
        The curves are loaded in the same order that are in flist. 
    
    data_arr : np.array
        Numpy array containing the scattering vector (q) and scattering intensity (i).
        Array has shape (n, r), where n is the number of scattering curves loaded and r is the 
        number of entries in each loaded curve. 2 represents scattering intensity(i), for which 
        there are r number of values in each, for each loaded curve n. The curves are loaded in 
        the same order that are in flist.
    
    q :  np.array
        Numpy array containing scattering vector (q) values. 
    
    error : np.array
        Numpy array containing the experimental error for scattering intensity (i). Will be an 
        empty array if there is no error column in the imported 
        
    Raises:
    -------
    TypeError
        When flist or delim is not str type, mask is not int type, or err is not bool type. 
        
    IndexError
        When a column for error values is indicated but does not exist in the given files. 
        
    ValueError
        When the given delim does not match the delimitter in the given file. 
        

    Examples:
    ----------
    data, data_arr, q, error = load_set(flist=files, delim=' ', mask=12, err=False)
    
    for c in data_arr:
        plt.plot(q, c)
    
    '''
    
    # initialize list to store data
    data = []
    error = []
    
    # test if input parameters are proper data type
    try:
        if not isinstance(flist, list):     
            raise TypeError('\033[1;91mTypeError: Oops! flist must be list type. Try again...\033[1;91m')
            
        if not isinstance(delim, str):     
            raise TypeError('\033[1;91mTypeError: Oops! delim must be str type. Try again...\033[1;91m')
        
        if not isinstance(mask, int):     
            raise TypeError('\033[1;91mTypeError: Oops! mask must be int type. Try again...\033[1;91m')
        
        if not isinstance(err, bool):     
            raise TypeError('\033[1;91mOops! err must be bool type. Try again...\033[1;91m')
            
                    
        # load laser on data
        for f in tqdm(flist, desc='Loading curves'):
            curve = load_saxs(file=f, delim=delim, mask=mask) 
        
            # append curve to vector list
            data.append(curve[:,1])
        
            if err == True:
                if curve.shape[1] < 3:
                    raise IndexError('\033[1;91mFile does not contain errors. Change err to False\033[1;91m')
                else:
                    error.append(curve[:,2])
            
            else:
                continue 

            
    except Exception as e:
        print(e.args[0])
        if isinstance(e, TypeError):
            if e.args[0][17] == 'f':
                flist = input('TypeError: Enter a new list value for flist. Include full path and quotations for each file name (do not include quotations around list): ')
                
            elif e.args[0][17] == 'd':
                delim = input('TypeError: Enter a new str value for delim (do not include quotations): ')
              
            elif e.args[0][11] == 'm':
                mask = int(input('TypeError: Enter a new int value for mask: '))  
                
            elif e.args[0][17] == 'e':
                err = input('TypeError: Enter a new bool value for err: ')
                
        elif isinstance(e, ValueError):
            delim = str(input('ValueError: File has a different delim. Enter a new str value for delim (do not include quotations): '))
            
        elif isinstance(e, IndexError):
            print('\033[1mIndexError: err=True indicates file contains error but there is no column containing errors. Changing err to False...')
            err = False
            # load laser on data
            
            for f in tqdm(flist, desc='Loading curves'):
                curve = load_saxs(file=f, delim=delim, mask=mask) 
        
                # append curve to vector list
                data.append(curve[:,1])
            data.remove(data[0])
           

    # turn data list into array
    data_arr = np.array(data)
    
    # turn err list into array
    error = np.array(error)
    
    # get q
    q = curve[:,0]
    
    print('\033[1;92mDone loading ' + str(len(data)) + ' curves!\033[1;92m')
        
    return data, data_arr, q, error
