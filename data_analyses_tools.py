import numpy as np
import os
import pandas as pd


def write_to_file(fname, new_data, update_duplicate=False, duplicate_col=''):
    """
        Writes a pandas df *new_data* to a tab separated text file *fname*, or 
        appends the data to the file. 
        Optionally it can also check whether the new_data is already present, by 
        comparing the new_data in column *duplicate_col*.
        
        Parameters
        ----------
        fname : str
            Name of the file.
        new_data : pandas DataFrame
            The new_data to append to *fname*
        update_duplicates : boolean
            Update if the current *new_data* already exist in column 
            *duplicate_col*
        duplicate_col : str
            Name of the column to check for duplicate values
    """
    # Check if file already exists and update existing data     
    if os.path.isfile(fname) and update_duplicate:
        with open(fname, 'r') as f:
            old_data = pd.read_csv(f, header=0, sep='\t')
        duplicate=old_data[duplicate_col].isin(new_data[duplicate_col])    
        if duplicate.any():
            old_data.loc[duplicate, :] = new_data.values
        else:
            old_data=old_data.append(new_data)
        with open(fname, 'w') as f:
            old_data.to_csv(f, sep='\t', index=False)    
    # Check if file already exists and append data     
    elif os.path.isfile(fname) and (not update_duplicate):        
        with open(fname, 'a') as f:
            new_data.to_csv(f, sep='\t', index=False, header=False)
    # If no file exists, write data to new file        
    else:
        new_data.to_csv(fname, sep='\t', index=False)   


def conv_3D(u, v, w):
    """ Converts data so it can be plotted in 3D graphs.

        Parameters
        ----------
        u : 1d pandas df
            Data should vary between two extremes, while v is constant.
        v : 1d pandas df
            Data in v is constant for 1 u sweep, and then changes.
        w : 1d pandas df
            Data measured as a function of u and v.

        Returns
        -------
        X, Y, Z : 2D arrays
            Data suitable for 3D plotting.
    """
    # Make new variables for 3D plot
    X = np.linspace(min(u), max(u), num=u.idxmax(axis=1)+1)
    Y = np.linspace(min(v), max(v), num=len(v)/len(X))
    X, Y = np.meshgrid(X, Y)
    Z = np.ndarray(np.shape(X))
    for i in range(len(X)):
        for j in range(len(X[0])):
            Z[i][j] = w.iloc[j+i*len(X[0])]
    return X, Y, Z


def deriv_clean(df, col):
    """ Removes the rows in a dataframe containing the highest and lowest value
        stored in V.

        Parameters
        ----------
        df : pandas dataframe
            Dataframe to be cleaned.
        col : str
            Column to look for the min and max value.

        Returns
        -------
        df : pandas dataframe
            Dataframe without a few rows.
    """
    # Remove first and last data points, because derivative there is not
    # correct.
    df = df[(df[col] > min(df[col])) & (df[col] < max(df[col]))]
    df = df.reset_index(drop=True)
    return df


def fix_time_stamps(time, meas_points):
    """ Fixes each first time stamp of a new LabView measurement. Also with the
        start of eacht new sweep, when doing a 2D measurement

        Parameters
        ----------
        time : pandas df, or array
            Time stamps of the measurement
        meas_points : int
            Amount of data points of one measurement. In case of a 1D scan this
            is equal to the len(time)

        Returns
        -------
        Array where the time stamps have been fixed.
    """
    # Convert to an array just the make sure pandas dataframes do not give
    # SettingWithCopyWarning
    time = np.array(time)
    delta_t = time[2] - time[1]
    for i in range(0, len(time), meas_points):
        time[i] = time[i+1] - delta_t
    return time


def make_folder(name):
    """ Checks if folder name exists, else it makes a new folder in the current
        directory

        Parameters
        ----------
        name : str
            Folder name

        Returns
        -------
        None
    """
    if not os.path.isdir(name):
        os.makedirs(name)


def remove_outliers(df, col, std=3):
    """ Removes outliers from the data.

        Parameters
        ----------
        df : pandas dataframe
            Data to clean up
        col : str
            Column to clean up
        std : int
            All data within this standard deviation is kept

        Returns
        ------
        df : pandas dataframe
            Cleaned dataframe
    """
    df = df[np.abs(df[col]-df[col].mean()) <= std*df[col].std()]
    return df


def rename(df, names, raise_error=True):
    """ Rename the column names of a pandas data frame. Raises an

    Parameters
    ----------
    df : pandas df
        The dataframe of which the column names will be changed
    names : dict
        Dictionairy with the old names and new names.
    raise_error : boolean (default = True)
        Raise an error when a certain variable name is not available.

    Returns
    -------
    df : pandas df
        Returns renamed df.
    """

    for k in names.keys():
        if not any(df.columns == k) and raise_error:
            raise NameError("Variable " + k + " not available")
        elif not any(df.columns == k):
            pass
        else:
            df.rename(columns={k: names[k]}, inplace=True)
    return df


def len_scan(A):
    """ Counts the number of data points at a constant value.

        A = array with values which repeat l number of times
    """
    for a in range(1, len(A)):
        if A[a] != A[a-1]:
            break
    return a


def tr_rt(d):
    """ Counts the number of data points in a trace+retrace measurement

        input:
        d = array with the x values of the measurement
    """
    return int((d[0]-d[1])*2*d[0]+1)


def trace_points(data):
    """ Counts the number of data points in a trace.

        Parameters
        ----------
        data : array, pandas df
            Values of the trace-retrace sweeps

        Returns
        -------
        points : int
            Amount of points of a trace
    """
    data = np.asarray(data)

    for i in xrange(len(data)):
        if np.sign(data[i+1]-data[i]) != np.sign(data[i+2]-data[i+1]):
            break
    return i+1


def chunks(l, n):
    """ Yield successive n-sized chunks from l.

        l = array to be divided into pieces
        n = size of the chunks
    """
    A = l[:n]
    if len(l) > n:
        for i in range(n, len(l), n):
            A = np.vstack((A, l[i:i+n]))
    return A


def achunks(l, n):
    """ Yield successive n-sized chunks from l, alternating the list
    order when n is uneven or even. Used for trace/rerace measurements.
    ex. [2,1,0,-1,-2,-2,-1,0,1,2] --> [[2,1,0,-1,-2],[2,1,0,-1,-2]]

        l = array to be divided into pieces
        n = size of the chunks
    """
    A = l[:n]
    for idx, i in enumerate(range(n, len(l), n)):
        if idx % 2 == 1:
            A = np.vstack((A, l[i:i+n]))
        if idx % 2 == 0:
            A = np.vstack((A, l[i+n-1:i-1:-1]))
    return A


def exp(x, a, b, x0, y0):
    """ Exponential decay with an y and x offset

        x = x variable (float)
        a = amplitude (float)
        b = decay parameters (float)
        x0 = x-offset (float)
        y0 = y-offset (float)
    """
    return a*np.exp(-b*(x-x0)) + y0


def average(x, y):
    """ Average y at common x values.

    Parameters
    ----------
    x, y: array, pandas dataframe

    Returns
    -------
    x_avg, y_avg : array
        Averaged arrays
    """
    x, y = np.asarray(x), np.asarray(y)
    x_avg = np.unique(x)
    y_avg = []
    for i in x_avg:
        y_avg.append(float(y[x == i].mean()))
    y_avg = np.asarray(y_avg)

    return x_avg, y_avg


def trace(x, y, sign=1):
    """ Extract trace (default sign=1) or retrace (sign=-1) from y given a x
        which e.g. goes back and forth between two extremes.

        Parameters
        ----------
        x, y : array, pandas dataframe

        Returns
        -------
        x_t, y_t : array
            Only the trace or retrace
    """
    x, y = np.asarray(x), np.asarray(y)
    if sign == 1:
        x_t = x[np.sign(np.gradient(x)) >= 0]
        y_t = y[np.sign(np.gradient(x)) >= 0]
    elif sign == -1:
        x_t = x[np.sign(np.gradient(x)) <= 0]
        y_t = y[np.sign(np.gradient(x)) <= 0]
    else:
        raise ValueError('No valid input for sign, must be 1 or -1')
    return x_t, y_t


def represents_int(s):
    """ Find out if a string is an integer.
        From: Triptych (stackoverflow)
    """

    try:
        int(s)
        return True
    except ValueError:
        return False


def find_header(filename, skip_blank_lines=True):
    """ Find the line number where the header of the LabView data. By default it
        does not counts blank lines (i.e. '\n').

        Parameters
        ----------
        filename : str
            Filename
        skip_blank_lines : boolean (default = True)
            Skip the blank lines when counting

        Returns
        -------
        count : int
            The amount of header lines.

    """
    with open(filename) as f:
        content = f.readlines()
    i = 0
    for c in content:
        if represents_int(c[0]):
            break
        if c == '\n':
            pass
        else:
            i += 1
    return i-1


def dim_LV_meas(filename, c=8, l=1):
    """ Check wheter a LabView measurement was a 1D or 2D measurement by
        checking if there is a X1 or X2 in line l.

    Parameters
    ----------
    filename : str
        Filename
    c : int (default = 8)
        Which character to look for a 1 or 2
    l : int (default = 1)
        Which line number contains X1 or X2

    Returns
    -------
    Returns a 1 for a 1D measurement or a 2 for a 2D measurement.
    """
    with open(filename) as f:
        content = f.readlines()
    dim = content[l][c]
    if not represents_int(dim) and (dim != 1 or dim != 2):
        raise ValueError('Can not determine measurement dimension from '
                         'character ' + str(c) + ' in line ' + str(l) +
                         ' from ' + filename)
    return int(content[l][c])


def index_char(s, c):
    """ Returns an array with each position of *c* in *s*.

        Parameters
        ----------
        s : string
            String to search for *c*
        c : char
            Character to find in *s*
    """
    return [index for index, value in enumerate(s) if value == c]


def SV_B_sweep(B_max, B_step, t=1, B_align=30, fname='SV_sweep.txt'):
    """
    Create a mangetic field sweep for a spin valve measurement.

    First column contains *B* sweeps which goes from -*B_align* to 0 and then
    to *B_max* in *step* size, then to *B_align* to 0 and to -*B_max*. Second
    column contains the wait time at each *B* value.

    Parameters
    ----------
    B_max : float
        The largest B value.
    B_step : float
        Magnetic field step size.
    t : float (default = 1s)
        Time to wait at each field value.
    B_align : float (default = 30)
        The field to align the electrodes.
    fname : str (default='SV_sweep.txt')
        Filename of the B sweep.

    Returns
    -------
    Saves the B sweep in a tab separated file.

    Example
    -------
    >>> SV_B_sweep(4, 2)
    Saves 'SV_sweep.txt' containing:
        [[-30, 0, 2, 4, 30, 0, -2, -4], [1, 1, 1, 1, 1, 1, 1, 1]]
    """
    B_sweep = np.hstack([[-1*B_align], np.arange(0, B_max+B_step, B_step)])
    B_sweep = np.hstack([B_sweep, B_sweep*-1])
    df = pd.DataFrame()
    df['B'] = B_sweep
    df['time'] = t
    df.to_csv(fname, header=False, index=False, sep='\t')
    print('Sweep saved in file: ' + fname)


def Hanle_B_sweep(B_max, B_step, B_small=0, B_small_step=0, t=1,
                  fname='Hanle_sweep.txt'):
    """
    Create a mangetic field sweep for a Hanle measurement.

    First column contains *B* sweeps which goes from -*B_max* to *B_max* in
    *B_step*, except if B_small is set. Then in the region from -*B_small* to
    *B_small*, where it will do *B_small_step*. Second column contains the wait
    time at each *B* value.

    Parameters
    ----------
    B_max : float
        The largest B value in Ampere.
    B_step : float
        Magnetic field step size in Ampere.
    B_small : float (default = 0)
        Field value where to start a smaller step size, shoudl be lower than
        *B_max*.
    B_small_step : float (default = 0)
        The step size at smaller field values, starting at *B_small*.
    t : float (default = 1s)
        Time to wait at each field value in seconds.
    fname : str (default='Hanle_sweep.txt')
        Filename of the B sweep.

    Returns
    -------
    Saves the B sweep in a tab separated file.

    Example
    -------
    >>> Hanle_B_sweep(10.0, 2.0, 2.0, 0.5)
    Saves 'Hanle_sweep.txt' containing:
        [[-10, -8, -6, -4, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 4, 6, 8, 10],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    """

    B_sweep = np.arange(-B_max, B_max + B_step, B_step)
    if B_small != 0:
        B_sweep = B_sweep[abs(B_sweep) > B_small]
        B_sweep = np.insert(B_sweep, len(B_sweep) / 2,
                            np.arange(-B_small, B_small + B_small_step,
                                      B_small_step))
    df = pd.DataFrame()
    df['B'] = B_sweep
    df['time'] = t
    df.to_csv(fname, header=False, index=False, sep='\t')
    print('Sweep saved in file: ' + fname)


def process_NL(dname, I, Vgain, cname):
    """
    Process non local measurement data from LabView and save it in a file.

    Calculate the mangetic field in Tesla and the non-local resistance.
    Additionaly it removes the two data points at the highest field values;
    especially convienient for spin valve measurements, not important for Hanle
    measurements.

    Parameters
    ----------
    dname : str
        Filename of the data.
    I : float
        Current used during the measurement
    Vgain : array
        The gain used on the IV meetkast for the different LIs.
    cname : str
        Calibration filename.

    Returns
    -------
    Saves the processed data in *dname* + '_proc.dat'
    """
    df = pd.read_table(dname)
    df = rename(df, {'X1': 'B(A)', 'Y1(X)': 'V1', 'Y2(X)': 'V2',
                     'Y3(X)': 'V3'}, raise_error=False)
    B_cal = pd.read_table(cname, skiprows=5, header=None,
                          names=['B(A)', 'B(mT)'])
    df['B(mT)'] = np.interp(df['B(A)'], B_cal['B(A)'], B_cal['B(mT)'])
    for i, v in enumerate(Vgain):
        df['Rnl' + str(i+1)] = df['V' + str(i+1)]/(I*v)
    # Remove high field alignment of FMs
    df = df[np.abs(df['B(A)']) < max(df['B(A)'])]
    df.to_csv(dname[:-4] + '_proc.dat', sep='\t', index=False)
