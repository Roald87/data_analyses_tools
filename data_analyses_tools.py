import numpy as np
import os


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

        Return
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

        Return
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
                         'character ' + str(8) + ' in line ' + str(l) +
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
