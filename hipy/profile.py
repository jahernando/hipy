#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:25:22 2022

@author: hernando
"""


import numpy as np

from   collections import namedtuple
from   scipy       import stats


import matplotlib.pyplot as plt
import hipy.pltext       as pltext


Profile = namedtuple('Profile', 
                     ('counts', 'mean', 'std', 'chi2', 'pvalue', 'success',
                      'bin_centers', 'bin_edges', 'bin_indices', 'residuals'))

def _residuals(values, mean, std, ibins):
    
    def _res(val, idx):
        _res  = val - mean[idx]
        _std  = float(std[idx])
        if _std > 0: _res = _res/_std
        return _res
    zbins = ibins if len(ibins) == len(values) else zip(*ibins)
    res = [_res(val, idx) for val, idx in zip(values, zbins)]
    return np.array(res, float)
 
def _correction(values, mean, ibins, scale = 1.):
 
    def _coor(val, idx):
        factor = mean[idx]
        factor = scale / factor if factor > 0 else 0.
        return factor * val

    zbins = ibins if len(ibins) == len(values) else zip(*ibins)
    corvalues = [_coor(val, idx) for val, idx in zip(values, zbins)]
    return np.array(corvalues, float)


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
    Profile     : (counts, mean, std, chi2, p-value, 
                   bin centers, bin edges, bin indices, residuals)
        bin-indices : for every entry of weights the index of its bin
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
    
    return Profile(counts, mean, std, chi2, pval, success, cbins, ebins, ibins, res)


def profile_scale(coors, weights, profile, scale = 1.):
    """
    
    Apply corrections from a profile to the weights

    Parameters
    ----------
    coors   : tuple(np.array), list of the values of the coordinates, i.e (x, y, z)
    weights : np.array, values of the weights 
    profile : Profile, profile named tuple
    x0      : float, scale

    Returns
    -------
    cor_weights : np.array, corrected weights
    """
    
    mean  = profile.mean
    #ebins = profile.bin_edges
    ibins = profile.bin_indices

    # _, _, ibins = stats.binned_statistic_dd(coors, weights, 
    #                                         bins = ebins, statistic = 'count',
    #                                         expand_binnumbers = True)
    # ibins = [b-1 for b in ibins]
    
    cor_weights = _correction(weights, mean, ibins, scale)
    
    return cor_weights
    

def plot_profile(profile, nbins = 50, stats = 'all', coornames = ('x', 'y', 'z')):
    """
    Plot profile
    """
    
    #counts = profile.counts
    #mean   = profile.mean
    #std    = profile.std
    #chi2   = profile.chi2
    #pval   = profile.pvalue
    cbins  = profile.bin_centers
    ebins  = profile.bin_edges
    mask   = profile.success
    #res    = profile.residuals

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