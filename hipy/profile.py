#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:25:22 2022

@author: hernando
"""


import numpy  as np
import pandas as pd

from   collections import namedtuple
from   scipy       import stats


import matplotlib.pyplot as plt
import hipy.pltext       as pltext


profile_names = ('counts', 'mean', 'std', 'chi2', 'pvalue', 'success',
                 'bin_centers', 'bin_edges')
Profile        = namedtuple('Profile', profile_names)
                     
def _residuals(values, mean, std, ibins):
    
    def _res(val, idx):
        _res  = val - mean[idx]
        _std  = float(std[idx])
        if _std > 0: _res = _res/_std
        return _res
    zbins = ibins if len(ibins) == len(values) else zip(*ibins)
    res = [_res(val, idx) for val, idx in zip(values, zbins)]
    return np.array(res, float)
 

def profile(coors, weights , bins = 20, counts_min = 3):
    """
    

    Parameters
    ----------
    coors       : tuple(np.array), list of the values of coordinates, i.e (x, y, z)
    weights     : np.array, values of the weights
    bins        : int, typle(int), or typle(np.array), bins of the profile.
                  Default = 20 in each dimension
    counts_min  : int, minimum counts in bin to compute the p-value. 
                  Default = 3

    Returns
    -------
    Profile     : (counts, mean, std, chi2, p-value, success, 
                   bin centers, bin edges)
    residual    : for every weifghts, the residual associated to that bin, that is:
                     (weight - mean[i]) / std[i], where i is the index of the bin 
                     assocaited to the weight.
    """
    
    counts, ebins, ibins = stats.binned_statistic_dd(coors, weights, 
                                                     bins = bins, statistic = 'count',
                                                     expand_binnumbers = True)    
    ibins = [b-1 for b in ibins]

    mean, _, _  = stats.binned_statistic_dd(coors, weights, bins = bins, statistic = 'mean')
    
    std, _ , _  = stats.binned_statistic_dd(coors, weights, bins = bins, statistic = 'std')

    res         = _residuals(weights, mean, std, ibins)    
    
    chi2, _ , _ = stats.binned_statistic_dd(coors, res * res, bins = bins, statistic = 'sum')
    #sf          = stats.chi2.sf(chi2, counts)
    
    pvalue      = lambda x : stats.shapiro(x)[1] if (len(x) > counts_min) else 0.
    pval, _, _  = stats.binned_statistic_dd(coors, weights, bins = bins, statistic = pvalue)
        
    success     = counts > counts_min
    
    cbins       = [0.5 * (x[1:] + x[:-1]) for x in ebins]
    
    return Profile(counts, mean, std, chi2, pval, success, cbins, ebins), res



def profile_scale(coors, weights, profile, scale = 1., mask = None):
    """
    
    Apply corrections from a profile to the weights

    Parameters
    ----------
    coors   : tuple(np.array), list of the values of the coordinates, i.e (x, y, z)
    weights : np.array, values of the weights 
    profile : Profile, profile named tuple
    scale   : float, scale

    Returns
    -------
    cor_weights : np.array, corrected weights
    """
    
    ndim      = len(coors)
    bin_edges = profile.bin_edges
    
    idx  = [np.digitize(coors[i], bin_edges[i])-1          for i in range(ndim)]
    sels = [(idx[i] >= 0) & (idx[i] < len(bin_edges[i])-1) for i in range(ndim)]
    sel  = sels[0]
    for isel in sels[1:]: sel = np.logical_and(sel, isel)

    idx    = tuple([idx[i][sel] for i in range(ndim)])
    
    ene    = weights[sel] 
    
    mean   = profile.mean
    mask   = profile.success if mask == None else mask
    
    mean[~mask] = np.nan
    
    mean   = mean[idx]

    vals   = scale * ene / mean
    
    cor_weights              = np.nan * np.ones(len(weights))
    cor_weights[sel == True] = vals

    return cor_weights
    
    

def save(profile, key, ofilename):
    """
    
    store the profile into a hdf5 outputfile as a DataFrame

    Parameters
    ----------
    profile   : Profile, profile to store
    key       : str, name of the profile, used as key in the hdf5 file
    ofilename : str, output file name

    Returns
    -------
    None.

    """
    
    odf = {}
    names = profile._fields[:-2]
    for name in names: odf[name] = getattr(profile, name).ravel()
    odf = pd.DataFrame(odf)

    obins = {}    
    bins  = profile.bin_edges
    for i, b in enumerate(bins): obins[i] = b
    obins = pd.DataFrame(obins)
    
    odf  .to_hdf(ofilename, key = key + '/profile', mode = 'a')
    obins.to_hdf(ofilename, key = key + '/bins'   , mode = 'a')
    
    return


def load(key, ifilename, type = Profile):
    """
    
    load a Profile (or other similar object) from a input file

    Parameters
    ----------
    key       : str, name of the profile, key of the data frame
    ifilename : str, name of the inputfile
    type      : Construction, optional, Default is Profile

    Returns
    -------
    prof     : object, profile

    """
    
    df     = pd.read_hdf(ifilename, key = key + '/profile')
    dfbins = pd.read_hdf(ifilename, key = key + '/bins')

    nbins       = len(dfbins.columns)
    bin_edges   = [dfbins[i].values for i in range(nbins)]
    bin_centers = [0.5*(b[1:] + b[:-1]) for b in bin_edges]
    
    shape  = tuple([len(b)-1 for b in bin_edges])

    names = tuple(df.columns)    
    print(names)
    vars   = [df[name].values.reshape(shape) for name in names]

    prof = type(*vars, bin_centers, bin_edges)
    
    return prof
    

#---- Plotting

def plot_profile(profile, nbins = 50, stats = 'all', coornames = ('x', 'y', 'z')):
    """
    
    plot the profile 

    Parameters
    ----------
    profile   : Object, profile
    nbins     : int, nbins, default = 50
    stats     : str or tupe(str), list of the variables to plot, default 'all'
    coornames : type(str), name of the coordinates of the profile

    Returns
    -------
    None.

    """
    
    cbins  = profile.bin_centers
    ebins  = profile.bin_edges
    mask   = profile.success

    def _var1(var, title):
        canvas = pltext.canvas(2, 2)
        canvas(1)
        pltext.hist(cbins[0][mask], bins = ebins[0], weights = var[mask], stats = False);
        name = coornames[0]
        plt.xlabel(name); plt.ylabel(title);
        canvas(2)
        uvar = np.nan_to_num(var, 0.)
        pltext.hist(uvar[mask], nbins);
        plt.xlabel(title)
        plt.tight_layout();
        return
        
    def _var2(var, title):
        mesh   = np.meshgrid(cbins[0], cbins[1])
        canvas = pltext.canvas(2, 2)
        canvas(1)
        uvar = np.copy(var)
        if (uvar.dtype != bool): uvar[~mask] = np.nan
        plt.hist2d(mesh[0].ravel(), mesh[1].ravel(), bins = ebins, 
                   weights = uvar.T.ravel());
        xname, yname = coornames[0], coornames[1]
        plt.xlabel(xname); plt.ylabel(yname); plt.title(title);
        plt.colorbar();
        canvas(2)
        pltext.hist(var[mask].ravel(), nbins);
        plt.xlabel(title)
        plt.tight_layout();
        return
        
    def _var3(uvar, title):
        mesh   = np.meshgrid(cbins[0], cbins[1])
        for i in range(len(cbins[-1])):
            var   = uvar[:, :, i]
            imask = mask[:, :, i] 
            canvas = pltext.canvas(2, 2)
            canvas(1)
            vvar = np.copy(var)
            if (vvar.dtype != bool): vvar[~imask] = np.nan
            plt.hist2d(mesh[0].ravel(), mesh[1].ravel(), bins = ebins[:-1], 
                       weights = vvar.T.ravel());
            xname, yname, zname = coornames[0], coornames[1], coornames[2]
            plt.xlabel(xname); plt.ylabel(yname); 
            plt.title(title + ', {:s} = {:4.2f} '.format(zname, cbins[-1][i]));
            plt.colorbar();
            canvas(2)
            pltext.hist(var[imask].ravel(), nbins);
            plt.xlabel(title)
            plt.tight_layout();
        return

    _vars = {1: _var1, 2: _var2, 3: _var3}
    ndim = len(profile.counts.shape)
    _var = _vars[ndim]
        

    stats = ('counts', 'mean', 'std', 'chi2', 'pvalue') if stats == 'all' else stats
    
    for name in stats:
        
        var = getattr(profile, name)
        
        _var(var, name)