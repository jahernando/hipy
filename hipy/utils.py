"""
    Some utilities for numpy and padas.
"""

import numpy   as np
import pandas  as pd


#--- general utilies

def remove_nan(vals : np.array) -> np.array:
    """
    return the values of the array that are not NaN

    Parameters
    ----------
    vals : np.array

    Returns
    -------
    vals : np.array (without nans)

    """
    return vals[~np.isnan(vals)]


def in_range(values : np.array, range : tuple = None, upper_limit_in = False) -> np.array:
    """
    classify the entries of values inside or not in the range.

    Parameters
    ----------
    values : np.array
    range : tuple, valid range. The default is None.
    upper_limit_in : bool, upper extreme is < or <=. The default is False (<).

    Returns
    -------
    sel   : np.array(bool), same size as values, and True/False if the value is inside the range

    """
    if (range is None): return values >= np.min(values)
    sel1 = (values >= range[0])
    sel2 = (values <= range[1]) if upper_limit_in else (values < range[1])
    sel  = (sel1) & (sel2)
    return sel


def centers(xs : np.array) -> np.array:
    """
    Returns the centers of an array,

    Parameters
    ----------
    xs : np.array

    Returns
    -------
    centers : np.array

    """
    return 0.5 * ( xs[1: ] + xs[: -1])


def yerrors(y      : np.array, 
            minval : float = 1.4) -> np.array:
    
    return np.maximum(minval, np.sqrt(y))


# def edges(x : np.array) -> np.array:
    
#     xc = centers(x)
#     x0 = x[:-1] + 0.5
#     x1 = 2 * xc[-1] - xc[-2]
#     xe = np.concatenate(((x0,), xc, (x1,)))
#     return xe
                        
    


def arscale(x     : np.array,
            scale : float = 1.):
    """
    
    rscale the array between [0, scale]

    Parameters
    ----------
    x     : np.array
    scale : float, optional, maximum value of the scale
         The default is 1.

    Returns
    -------
    rx    : np.array

    """
    
    xmin, xmax = np.min(x), np.max(x)
    rx = scale * (x - xmin)/(xmax - xmin)
    
    return rx
#
#
# def arstep(x, step, delta = False):
#     """ returns an array with bins of step size from x-min to x-max (inclusive)
#     inputs:
#         x    : np.array
#         step : float, step-size
#     returns:
#         np.array with the bins with step size
#     """
#     delta = step/2 if delta else 0.
#     return np.arange(np.min(x) - delta, np.max(x) + step + delta, step)


def stats(x : np.array, weights : np.array = None):
    """
    compute the size, average and standad deviation of 

    Parameters
    ----------
    x       : nparray, 
    weights : np.array, optional. The default is None.

    Returns
    -------
    size : float, sum of the weights, or total entries of x
    ave  : float, average of x
    std  : float, standard deviation of x
    """
    weights = np.ones(len(x)) if weights is None else weights
    assert len(weights) == len(x), 'stats requires same size of weights and x'
    size    = np.sum(weights)
    weights = weights / size
    ave     = np.average(x           , weights = weights)
    std     = np.sqrt(np.average((x - ave)**2, weights = weights))       
    return size, ave, std
    

#-----

def efficiency(selection : np.array, norma : int = None) -> tuple:
    """
    compute the efficiency and its error from a selection

    Parameters
    ----------
    selection : np.array(bool)
    norma : int, optional (the number of entries of selection)

    Returns
    -------
    eff  : float, efficiency
    ueff : float, efficiency uncertainty

    """
    norma = norma if norma is not None else len(selection)
    eff   = np.sum(selection)/norma
    ueff  = np.sqrt(eff * (1- eff) / norma)
    return eff, ueff


# def df_values(df    : pd.DataFrame,
#                names : tuple, 
#                sel   : np.array = None):
#     """
    
#     returns a list of np.arrays with the values of variables with names
#     that pass a selection

#     Parameters
#     ----------
#     df    : pd.DataFrame
#     names : tuple(str), list with the names of the variables
#     sel   : np.array(bool), selection of the rows of DF. 
#             Must be of the same length of df.
#             Default is None

#     Returns
#     -------
#     vals  : list(np.array), list with the arrays of the variables with names
#         that pass the selection

#     """
    
#     sel = np.ones(len(df), bool) if sel is None else sel
#     assert len(sel) == len(df), \
#         'Same length of the selection and the number of rows in DF required'
    
#     xdf  = df[sel]
#     vals = [xdf[name].values for name in names]
#     return vals


# def selection(df : pd.DataFrame, columns : tuple, ranges = tuple) -> np.array:
#     """
#     Returns an array with True/False is the row/index of the DataFrame has
#     the values of the columns inside its range.

#     Parameters
#     ----------
#     df      : pd.DataFrame,
#     columns : tuple(str), list with the names of the columns
#     ranges  : tuple(range), list with the ranges for each column

#     Returns
#     -------
#     selections: np.array(bool), list the the selection

#     """
#     sel =  None
#     for i, column in enumerate(columns):
#         isel = in_range(df[column], ranges[i])
#         sel  = isel if sel is None else sel & isel
#     return sel
