import numpy as np

import hipy.utils as ut
import hipy.cfit  as cfit

from scipy.stats import binned_statistic as _profile


"""
    Extension of mathematical histogram: fitting histogram, profiles
"""


def hfit(x     : np.array, 
         bins  : int, 
         range : tuple = None,
         fun   : str = 'gaus',
         p0    : np.array = None,
         **kargs) -> tuple:
    """


    Parameters
    ----------
    x    : np.array,
    bins : int, number of bins, of array with the bins.
    range: tuple, range. Default is None
    fun  : str or callable, function to fit. fun(x, *parameters). Default is 'gaus'
          Options suported: 'gaus', 'line', 'exp', 'gausline', 'gausexp'.
    guess : np.array, parameters of function. Default is None
    **kargs : dict, labeled arguments for np.histogram

    Returns
    -------
    yc    : np.array, contents of the histogram
    edges : np.array, bin edges of the histogram
    ye    : np.array, errors of contents
    pars  : np.array, list of the parameters of the fit.
    epars : np.array, uncertainties of the parameters.

    """

    yc, edges = np.histogram(x, bins, range)
    xc        = 0.5 * (edges[1:] + edges[:-1])
    ye        = np.maximum(2.4, np.sqrt(yc))
    
    pars, epars, ffun = cfit.curve_fit(xc, yc, p0 = p0, sigma = ye, 
                                       fun = fun, **kargs)

    return yc, edges, ye, pars, epars, ffun


def hprofile(x      : np.array,
             y      : np.array,
             bins   : int, 
             xrange : tuple = None,
             fun    : callable = None,
             percentile : bool = False):
    """
    
    Compute the profile of y vs x. Accept entries in the x and y ranges.
    Create partition in x-range with bins.
    Returns the counts, mean, average and error in the average in each x-bin.
    If there is no entries in a given bin, it returns nan.

    Parameters
    ----------
    x      : np.array
    y      : np.array
    bins   : int or np.array with the bin edges
    xrange : tuple, optional, range in x. The default is None.
    fun    : callable. optional. returns the atray fun(yi) yi, slice of i. 
             Default is None
    percentile: bool. optional. Create bins with equal number of counts. 
    
    Returns
    -------
    counts : np.array, counts in x-bins
    xmean  : np.array, mean average of x in x-slices
    xstd   : np.array, std of x in x-slices
    ymean  : np.array, y-mean in x-slices
    ystd   : np.array, y-std  in x-slices
    yfun   : np.array, fun(y_i). only if fun is provided
    """
        
    sel = ut.in_range(x, xrange)
    xp, yp = x[sel], y[sel] 
    
    if (percentile and type(bins == int)):
        bins = np.percentile(xp, np.linspace(0., 100., bins))
    
    counts, edges, ipos = _profile(xp, yp, 'count', bins, xrange)
    
    nbins = len(edges)
    xmean = np.array([np.mean(xp[ipos == i]) for i in range(1, nbins)])
    xstd  = np.array([np.std (xp[ipos == i]) for i in range(1, nbins)])
    ymean = np.array([np.mean(yp[ipos == i]) for i in range(1, nbins)])
    ystd  = np.array([np.std (yp[ipos == i]) for i in range(1, nbins)])
    
    res = (counts, xmean, xstd, ymean, ystd)

    if (fun is not None):
        yval = np.array([fun   (yp[ipos == i]) for i in range(1, nbins)])
        res  = *res, yval

    return res
    

# def hprofile(x : np.array, y: np.array, bins: int,
#              xrange : tuple = None, yrange : tuple = None):
#     """
    
#     Compute the profile of y vs x. Accept entries in the x and y ranges.
#     Create partition in x-range with bins.
#     Returns the counts, mean, average and error in the average in each x-bin.
#     If there is no entries in a given bin, it returns nan.

#     Parameters
#     ----------
#     x      : np.array
#     y      : np.array
#     bins   : int or np.array with the bin edges
#     xrange : tuple, optional, range in x. The default is None.
#     yrange : tuple, optional, range in y. The dafault is None.

#     Returns
#     -------
#     ysize  : np.array, counts in x-bins
#     xedges : np.array, edges of the x-bins
#     ymean  : np.array, y-mean in x-bins
#     ystd   : np.array, y-std  in x-bins
#     yumean : np.array, uncertainty in y-mean in x-bins
#     """

#     sel = (ut.in_range(x, xrange)) & (ut.in_range(y, yrange))
#     xp, yp = x[sel], y[sel] 
#     ysize, xedges = np.histogram(xp, bins = bins, range = xrange)
    
#     ipos = np.digitize(xp, xedges) - 1
    
#     nbins = len(xedges) -1
#     ymean  = np.array([np.mean(yp[ipos == i]) for i in range(nbins)])
#     ystd   = np.array([np.std (yp[ipos == i]) for i in range(nbins)])
#     yumean = ystd/np.sqrt(ysize)
 
#     return ysize, xedges, ymean, ystd, yumean
    


def in_nsigmas_of_profile(x       : np.array,
                          y       : np.array,
                          nbins   : int,
                          xrange  : tuple = None,
                          yrange  : tuple = None,
                          nsigmas : float = 2.,
                          niter   : int = 1):
    """
    
    returns the selection of the (x, y) that are in nsigmas inside the profile
    defined by the ranges (xrange, yrange) and nbins.

    Parameters
    ----------
    x       : np.array
    y       : np.array
    nbins   : int, number of x-bins of the profile
    xrange  : tuple, optional. x-range. Default is None.
    yrange  : tuple, optional. y-range. Default is None
    nsigmas : float, optional. number of sigmas inside the profile. Default is 2.
    niter   : int, optional. number of iterations. The default is 1.

    Returns
    -------
    sel     : np.array(bool). bool-array with True/False if (x, y) if in the selection.

    """
            
    sel  = ut.in_range(x, xrange) & ut.in_range(y, yrange)
    
    def _sel(x, y, xedges, ymed, ystd):
        ipos  = np.digitize(x, xedges) - 1
        ipos  = np.minimum(ipos, nbins - 1)
        ipos  = np.maximum(0, ipos)
        return abs(y - ymed[ipos]) / ystd[ipos] < nsigmas
    
    for i in range(niter):
        xp, yp = x[sel], y[sel]
        ysize, xedges, ymed, ystd, yumed = hprofile(xp, yp, nbins, xrange, yrange)
        sel = _sel(x, y, xedges, ymed, ystd)
        #print(i, np.sum(sel))
        
    return sel



    