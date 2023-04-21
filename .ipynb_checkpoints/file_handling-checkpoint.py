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
            raise TypeError('TypeError: file must be str type. Try again...')
            
        if not isinstance(delim, str):
            raise TypeError('TypeError: delim must be str type. Try again...')
        
        if not isinstance(mask, int):
            raise TypeError('TypeError: mask must be int type. Try again...')

                
                
        data = np.loadtxt(file, delimiter=delim, skiprows=mask)
                
                
                
    except Exception as e:
        tb = e.__traceback__
        print('\033[91m' + str(e.args[0]) + '\033[91m') 
        traceback.print_tb(tb)
        if isinstance(e, TypeError):
            if e.args[0][11] == 'f':
                print('\033[93mTypeError: Enter a new str value for file name including full path (do not include quotations): \033[93m')
                file = input()
                
            elif e.args[0][11] == 'd':
                print('\033[93mTypeError: Enter a new str value for delim (do not include quotations): \033[93m')
                delim = input()
              
            elif e.args[0][11] == 'm':
                print('\033[93mTypeError: Enter a new int value for mask: \033[93m')
                mask = int(input())
                
        elif isinstance(e, ValueError):
            print('\033[93mValueError: File has a different delim. Enter a new str value for delim (do not include quotations): \033[93m')
            delim = str(input())
        
    
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
            raise TypeError('TypeError: Oops! flist must be list type. Try again...')
            
        if not isinstance(delim, str):     
            raise TypeError('TypeError: Oops! delim must be str type. Try again...')
        
        if not isinstance(mask, int):     
            raise TypeError('TypeError: Oops! mask must be int type. Try again...')
        
        if not isinstance(err, bool):     
            raise TypeError('TypeError: Oops! err must be bool type. Try again...')
            
                    
        # load laser on data
        for f in tqdm(flist, desc='Loading curves'):
            curve = load_saxs(file=f, delim=delim, mask=mask) 
        
            # append curve to vector list
            data.append(curve[:,1])
        
            if err == True:
                if curve.shape[1] < 3:
                    raise IndexError('File does not contain errors. Change err to False')
                else:
                    error.append(curve[:,2])
            
            else:
                continue 

            
    except (TypeError, IndexError) as e:
        tb = e.__traceback__
        print('\033[1;91m' + str(e.args[0]) + '\033[1;91m')
        traceback.print_tb(tb)
        
        if isinstance(e, TypeError):
            if e.args[0][17] == 'f':
                print('\033[1;91mTypeError: Enter a new list value for flist. Include full path and quotations for each file name (do not include quotations around list): \033[1;91m')
                flist = input()
                
            elif e.args[0][17] == 'd':
                print('\033[1;91mTypeError: Enter a new str value for delim (do not include quotations): \033[1;91m')
                delim = input()
              
            elif e.args[0][11] == 'm':
                print('\033[1;91mTypeError: Enter a new int value for mask: \033[1;91m') 
                mask = int(input())
                
            elif e.args[0][17] == 'e':
                print('\033[1;91mTypeError: Enter a new bool value for err: \033[1;91m')
                err = input()
        elif isinstance(e, ValueError):
            print('\033[1;91mValueError: File has a different delim. Enter a new str value for delim (do not include quotations): \033[1;91m')
            delim = str(input())
            
        elif isinstance(e, IndexError):
            print('\033[1;91mIndexError: err=True indicates file contains error but there is no column containing errors. Changing err to False...\033[1;91m')
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

def plot_curve(data_arr, q_arr, labels=None, qmin=None, qmax=None,
               imin=None, imax=None, x='scattering vector',
               y='scattering intensity',
               title='SAXS Scattering Curve', save=False, 
               save_dir=None, save_name=None):
    '''
    Function to plot curve(s). Assumes input files are two columns, q
    and I. If save=True, then the plots will be saved to file in the 
    indicated save_dir with the file name save_name. If save_dir does
    not exist, then it will be automatically created. 
    
    Parameters:
    -----------
    data_arr : numpy array
        Numpy array containing scattering curves to be plotted. Must be provided. 
        Array can easily be generated with the `load_set` function. Assumes that
        the x values are loaded in a separate array called q, which can be easily 
        generated with the `load_set` function. 
        
    q_arr (optional) : numpy array
        Numpy array containing q values to be plotted on x-axis. Default value is q. 
        
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
        Label of plot y axis. Default value is 'scattering intensity'
    
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
    
    Raises:
    -------
    TypeError:
        data_arr or q are not np.ndarray data types. 
        
        
    Examples:
    ---------
    plot_curve(data_arr='data_arr', q_arr=q, labels=None, qmin=0.02, qmax=0.15, imin=None, imax=None, x='Scattering Vector', 
               y='Scattering Intensity', title='SAXS Scattering Curves', save=False, save_dir=None, save_name=None)
    
        
    '''
    
    # test if input parameters are proper data type
    try:
        if not isinstance(data_arr, np.ndarray):     
            raise TypeError('TypeError: Oops! data_arr must be np.ndarray type. Try again...')
            
        if not isinstance(q_arr, np.ndarray):     
            raise TypeError('TypeError: Oops! q_arr must be np.ndarray type. Try again...')
            
    except TypeError as e:
        tb = e.__traceback__
        print('\033[1;91m' + str(e.args[0]) + '\033[1;91m')
        traceback.print_tb(tb)
        
        if e.args[0][17] == 'd':
            print('\033[1;91mTypeError: Enter a new np.ndarray value for data_arr \033[1;91m')
            
        elif e.args[0][17] == 'q':
            print('\033[1;91mTypeError: Enter a new np.ndarray value for q_arr \033[1;91m')

        
    else:  
        # set color map
        n = len(data_arr)
        colors = pl.cm.rainbow_r(np.linspace(0,1,n))
    
        # plot data
        ax = plt.axes([0.125,0.125, 5, 5])
        for i,c in zip(data_arr, colors):
            plt.plot(q_arr, i, color=c)
        
        # set position for legend
        if qmin is None and qmax is None and imin is None and imax is None:
            leg_loc = (1.0, 0.5)
            leg_pos = 'center left'
            leg_cols = 5
        else:
            leg_loc = (-0.2, -0.08)
            leg_pos = 'upper center'
            leg_cols= 10
        
        # style plot    
        plt.legend(loc=leg_pos, bbox_to_anchor=leg_loc, fontsize=60, ncol=leg_cols, labels=labels)
        plt.xlabel(str(x), fontsize=60)
        plt.ylabel(str(y), fontsize=60)
        plt.title(str(title), fontsize=70)
        plt.xticks(fontsize=55)
        plt.yticks(fontsize=55)
        plt.set_cmap('viridis')

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(5)

        
        # define qmin and qmax
        if qmin is not None or qmax is not None:
            if qmin is not None and qmax is None:
                qmin = qmin
                qmax = np.max(data_arr[:,1])
    
            elif qmin is None and qmax is not None:
                qmin = np.min(data_arr[:,1])
                qmax = qmax
        
            elif qmin is not None and qmax is not None:
                qmin = qmin
                qmax = qmax
            
        # define imin and imax
        if imin is not None or imax is not None:
            if imin is not None and imax is None:
                imin = imin
                imax = np.max(data_arr[:,2])
    
            elif imin is None and imax is not None:
                imin = np.min(data_arr[:,2])
                imax = imax
        
            elif imin is not None and imax is not None:
                imin = imin
                imax = imax
            
        if qmin is not None or qmax is not None or imin is not None or imax is not None:
            #inset plot
            a = plt.axes([-5, 0.5, 4, 4])
        
            # loop over all curves
            for i,c in zip(data_arr, colors):
    
                # plot data
                plt.plot(q_arr, i, color=c)
        
            # style plot
            plt.xlabel(str(x), fontsize=60)
            plt.ylabel(str(y), fontsize=60)
            plt.xticks(fontsize=55)
            plt.yticks(fontsize=55)
            plt.xlim([qmin, qmax])
            plt.ylim([imin, imax])
            plt.title(str(title) + ' Zoom', fontsize=70)
            plt.set_cmap('viridis')

            for axis in ['top','bottom','left','right']:
                a.spines[axis].set_linewidth(5)

            # mark inset
            mark_inset(ax, a, loc1=1, loc2=4, fc="none", ec="0.5", linewidth=4)
            
        if save is True:
            make_dir(save_dir)
            plt.savefig(save_dir + save_name, bbox_inches='tight')
            
        plt.show()
            
        return