#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Module to create curves using a name (str)

Extend the curve_fit to functions defined given a name

Created on Thu Mar 25 09:17:39 2021

@author: hernando
"""


import numpy as np
import scipy.stats    as stats
import scipy.optimize as optimize

import operator  as operator
from functools import reduce

#import sys

#
#   curve fit
#----------------

def curve_fit(x     : np.array,
              y     : np.array,
              fun   : str or callable,
              p0    : np.array = None, 
              sigma : np.array = None,
              mask  : np.array = None, 
              **kargs):
    """
    Fit a curve y = fun(x, *p0), where p0 are the parameters of the curve
    
    Parameters
    ----------
    x   : np.array, x component of data
    y   : np.array. y component of data
    fun : str or callable, 
        if callable fun(x, y, *args), where *args are the arguments of the function to fit
        if str: the name of the function, 
            single functions  : 'norm', 'exp', 'polynomial.1', ...
            composed functions: 'exp+norm', ...
    p0     : np.array, 
        the initial parameter guess to start the fit
    sigma  : np.array, the errors on the y-coordinate, 
             if None, will consider statistical errors: max(sqrt(y), 2.4)
    mask   : np.array(bool),  
        mask any parameter in the fit. (True, False), the second parameters is not fitted
        The default is None. Fit all parameters
    **kargs : dict,
        key arguments for optimize.curve_fit method in scipy

    Returns
    -------
    pars  : np.array, fitted parameters
    upars : np.array, uncertainties of the fitted parameters
    fun   : callable, y = function(x) with the fitted parameters
    
    """
    
    if (mask is not None):
        assert (p0 is not None), 'If you mask please provide guess'
    
    ffun = curve(fun) if type(fun) == str else fun
    p0   = _curve_guess(fun)(x, y) if (p0 is None) and (type(fun) == str) else np.array(p0)
    
    
    mask = np.array(mask, bool) if mask is not None else None
    
    def place_mask(mask):
        
        mask  = np.array(mask, bool)
        par   = np.copy(p0)        
        assert len(mask) == len(p0), 'required same number of mask and parameters'
        mp0  = par[mask]    
        
        def mfun(x, *mpar):
            par[mask] = mpar
            return ffun(x, *par)
                
        return mfun, mp0
    
    cfun, cp0 = (ffun, p0) if mask is None else place_mask(mask)
        
    
    abs_sig = False if sigma is None else True
    pars, fcov = optimize.curve_fit(cfun, x, y, p0 = cp0, sigma = sigma, 
                                    absolute_sigma = abs_sig, **kargs)
    upars      = np.sqrt(np.diag(fcov))

    if (mask is not None):
        xpars        = np.copy(p0)    
        xpars[mask]  = pars
        pars         = xpars
        
        xupars       = np.zeros(len(p0))
        xupars[mask] = upars
        upars        = xupars
        
    
    xfun = lambda x : ffun(x, *pars)

    return pars, upars, xfun


def curve(name: str):
    """
    
    returns a curve, a function, y = fun(x, *pars), given a name.
    pars are the parameters of the curve

    Parameters
    ----------
    name : str
        name of the function.
        single names: 'norm' == 'gaus', 'exp', 'poynomial.1'
        composed    : 'exp+gaus', 'polynomial.1+gaus'
        valid names : the names of the stats function in scipy.stats, polynomial of numpy 

    Returns
    -------
    fun : callable
        the function: y = fun(x, *pars)
    """
    
    assert type(name) == str, 'no name for curve'
    
    name = 'norm' if name == 'gaus' else name
    
    if (name.find('+') > 0):
        names = name.split('+')
        return _curve_composite(names)

    if (hasattr(this_module.cfit, name)):
        return getattr(this_module.cfit, name)

    if (hasattr(stats, name)):
        return _curve_stats(name)
        
    if (name.find('.') > 0):
        name, deg =  name.split('.')
        return _curve_polynomial(name, int(deg))

    assert False, 'no valid name for curve : {:s}'.format(name)

    return None


def curve_argument_names(name: str):
    """
    
    return the names of the arguments of a curve given a name

    Parameters
    ----------
    name : str
        name of the curve, i.e. 'gaus', 'polynomial.1'

    Returns
    -------
    argnames: tuple(str)
        tuple with the names of the parameters of the function
        
    """
    
    if (name.find('+')>0):
        names = name.split('+')
        argnames = []
        for name in names:
            argnames += curve_argument_names(name)
        return argnames
    
    if (name in curve_argnames.keys()):
        return curve_argnames[name]
    
    if (hasattr(stats, name)):
        rv       = getattr(stats, name)
        argnames = ['loc', 'scale'] if rv.numargs <= 0 else rv.shapes
        return tuple(['size', ] + list(argnames))

    if (name.find('.') > 0):
        _, deg = name.split('.')
        argnames = [r'$a_'+str(i)+'$' for i in range(int(deg) + 1)]
        return tuple(argnames)
    
    assert False, 'no arguments names for curve : {:s}'.format(name)
    
    return None    
                              

#
#----- Internal Functions
#


#  Curve Functions Local Cathalogue
#---------------------------

#this_module = sys.modules[__name__]
this_module = __import__(__name__)
#curve_module = getattr(curve_module,'cfit') if hasattr(curve_module, 'cfit') else curve_module
#print(curve_module)

curve_argnames = {}
curve_argnames['gaus'] = (r'$N$', r'$\mu$', r'$\sigma$')
curve_argnames['exp']  = (r'$N$', r'$\tau$')

def exp(x, a, b):
    """ an exponential function a * exp(-b * x)
    """
    return a * np.exp( -  x / b )
    

def exp_guess(x, y):
    data = get_data(x, y)
    pars = stats.expon.fit(data)
    size = len(data)
    return np.array((size, pars[1]))


def get_data(x: np.array, y: np.array):
    ymin   = np.min(y[y > 0.])
    counts = np.array(y/ymin, int)
    xs     = np.concatenate([np.array(isize * (xi,)) for isize, xi in zip(counts, x)])
    return xs
        

# Curves by name
#---------------------

def _curve_composite(names : tuple):
    
     cvs    = [curve(name)                     for name in names]
     nargs  = [len(curve_argument_names(name)) for name in names]
     ntot   = np.sum(nargs)
     masks  = [np.zeros(ntot, bool)            for narg in nargs]
     for i, ni in enumerate(nargs):
         n0 = int(np.sum(nargs[:i]))
         masks[i][n0: n0 + ni] = True
            
     def _cv(x, *pars):   
         pars = np.array(pars)
         ys   = [cv(x, *pars[imask]) for cv, imask in zip(cvs, masks)]
         return reduce(operator.add, ys)
     
     return _cv    
    

def _curve_stats(name : str):

     rv    = getattr(stats, name)
     prob  = getattr(rv, 'pdf') if hasattr(rv, 'pdf') else None
     prob  = getattr(rv, 'pmf') if hasattr(rv, 'pmf') else prob

     def _cv(x, *pars):
        pars  = np.array(pars)
        norma = pars[0]
        upars = pars[1:]
        return norma * prob(x, *upars)
    
     return _cv


def _curve_polynomial(name: str, deg: int):
    name = 'Polynomial' if name.find('poly') >= 0  else name
    pol  = getattr(np.polynomial, name)
    
    def _cv(x, *args):
        return pol(args)(x)
    
    return _cv
                           
        
#  Guess Parameters
#--------------------


def _curve_guess(name: str):
 
    name = 'norm' if name == 'gaus' else name   
 
    if (name.find('+') > 0):
        names = name.split('+')
        return _curve_guess_composite(names)
     
    if (hasattr(this_module.cfit, name)):
        return getattr(this_module.cfit, name + '_guess')
     
    if (hasattr(stats, name)):        
        return _curve_guess_stats(name)
   
    if (name.find('.') > 0):
        name, deg = name.split('.')
        return _curve_guess_polynomial(name, deg)
    
    assert False, 'not possible to have initial guess for curve :{:s}'.format(name)

    return None    
    

def _curve_guess_composite(names: tuple):
    
    guess_cvs = [_curve_guess(name) for name in names]
    
    def _guess(x, y):
        pars = np.concatenate([iguess(x, y) for iguess in guess_cvs])
        return pars
    
    return _guess
    


def _curve_guess_stats(name : str):
    
    rv = getattr(stats, name)
    
    assert hasattr(rv, 'fit'), 'Please provide intial guess parameters for {:s}'.format(name)
    
    def _guess(x, y):
        data = get_data(x, y)
        size = len(data)
        pars = np.array((size, *rv.fit(data)))
        return pars    
    
    return _guess

    

def _curve_guess_polynomial(name : str, deg : int):
    
    name = 'Polynomial' if name.find('poly') >= 0 else name
    pol  = getattr(np.polynomial, name)
    
    def _guess(x, y):
        a = pol.fit(x, y, int(deg))
        return a.convert().coef
    
    return _guess
    
    