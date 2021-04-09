#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Module to create PDFs using a name (str)

Module to estimate the parameters of pdfs using MLL

Created on Thu Mar 25 09:17:39 2021

@author: hernando
"""

import numpy          as np
import scipy.stats    as stats
import scipy.optimize as optimize

import operator as operator
from functools import reduce


#--- Globals

current_module = __import__(__name__)

efit_argnames = {}
efit_argnames['gaus'] = (r'$\mu$', r'$\sigma$')
efit_argnames['exp']  = (r'$\tau$',)

method = 'Nelder-Mead'

#--- Main Functions


def mll(x       : np.array,
        fun     : str or callable, 
        p0      : np.array = None,
        mask    : np.array = None,
        sigma   : bool = False,
        epsilon : float = 0.1,  
        **kargs): 
    """
    
    Estimate the parameters of a pdf (fun) minimizing -2 log likehood

    Parameters
    ----------
    x : np.array
        data
    fun : str or callable
        if callable: log-likelihood function ll(x, *par)
        if str: name of the stats to the log-likelihood function
    p0 : np.array, optional
        initial parameters for the fit. In same cases are required.
        The default is None.
    mask : np.array(bool)
        mask of the parameters to fit.
        The default is None.
    sigma : bool, optional
        if bool is True, compute parameter uncertaities
        Notes:
            uses where 2 Delta LL = 1. 
            computes only one-side uncertainty.
            requires several minimization processes, it can be slow.
        The default is False.
    epsilon : float, optional
        (1+epsilon) * value: is the initial value of the parameter 
        when computing the uncertainties
        The default is 0.1.
    **kargs : dict,
        key arguments to minimize function of scipy.optimize

    Returns
    -------
    res : result of the minimize function (scipy.minimize)
        if sigma is True, the member 'sigma' (res.sigma) 
        contains the parameter uncertainties

    """
        
    ll   = stat_loglike(fun) if type(fun) == str else fun
    ffun = lambda pars: - 2 * ll(x, *pars)

    if (type(fun) == str):
        assert p0 is not None, 'Please enter initial parameters'
    p0  = _stat_guess(fun)(x) if p0 is None else p0
    p0  = np.array(p0)
    
    res = minimize(ffun, p0, mask = mask, **kargs)
    if (sigma       is False): return res
    if (res.success is False): return res

    # compute the uncertainties
    # consider symetrical uncertainties and obtained by Delta -2LL = 1 
    
    best  = res.x
    umask = np.ones(len(best), bool) if mask is None else np.copy(mask) 
    
    def _sqfun(index):
        imask    = np.copy(umask)
        imask[index] = False
        
        def _isqfun(v):
            p1        = np.copy(best)
            p1[index] = v
            kres      = minimize(ffun, p1, mask = imask)
            if (not kres.success): return 25.
            p1    = kres.x
            delta = ffun(p1) - ffun(best) -1
            return delta * delta    
        return _isqfun
            
    sigma   = []
    for i in range(len(best)):
        isigma = 0.
        if (umask[i]):
            v0    = (1. + epsilon) * best[i]
            sqfun = _sqfun(i)
            kres = optimize.minimize(sqfun, v0)
            isigma = np.abs(kres.x[0] - best[i])
        sigma.append(isigma)
    res.sigma = np.array(sigma)
    
    return res


def minimize(fun    : callable, 
             p0     : np.array, 
             mask   : np.array = None, 
             **kargs):
    """
    
    minimize function with parameters and a optional mask

    Parameters
    ----------
    fun : callable, 
        function to minimize, fun(par), for example -2 log likelihood
    pars : np.array,
        parameters
    mask : np.array(bool), optional.
        mask the parameters to fit.
        The default is None.
    **kargs : dict,
        key arguments for minimize
        
    Returns
    -------
    res : result from scipy.optimize.minimize.
          (res.x is the result of the fit, res.success can be True/False)
    """

    def place_mask(mask):
        
        mask  = np.array(mask, bool)
        par   = np.copy(p0)
        
        assert len(mask) == len(p0), 'required same number of mask and parameters'
        mp0 = par[mask]    
        
        def mfun(mpar):
            par[mask] = mpar
            return fun(par)
                
        return mfun, mp0

    cfun, cp0 = (fun, p0) if mask is None else place_mask(mask)
            
    if ('method' not in kargs.keys()): kargs['method'] = method
    
    res = optimize.minimize(cfun, cp0, **kargs)

    if (mask is not None):
        best = np.copy(p0)    
        best[mask] = res.x
        res.x = best

    return res


#
#  Create PDFs objects
#

def stat(name : str):
    """
    
    Return the stats-object given a name

    Parameters
    ----------
    name : str,
        Name of the stats-object
        simple    : 'norm' == 'gaus', 'exp', 'gamma', etc
        composite : 'exp+norm'
        
    Returns
    -------
    stat: obj for stats, with the methods: pdf(pmf), logpdf, rvs
    
    """
    
    name = 'norm' if name == 'gaus' else name
    
    if (name.find('+')>0):
        names = name.split('+')
        return CompositePDF(names)
    
    # efit
    if (hasattr(current_module.efit, name)):
        return getattr(current_module.efit, name)
    
    
    if (hasattr(stats, name)):
        return getattr(stats, name)
    
    assert False, 'No stat named : {:s}'.format(name)
    

def stat_loglike(name : str):
    """
    return the loglikelihood function given a name

    Parameters
    ----------
    name : str
        Name of the stats, i.e. 'gaus', 'exp+gaus'

    Returns
    -------
    ll : callable, ll(x, *pars)
        log-likelihood function
    """
    
    rv = stat(name)
    if (hasattr(rv, 'loglike')): return rv.loglike

    logprob = getattr(rv, 'logpdf') if hasattr(rv, 'logpdf') else None
    logprob = getattr(rv, 'logpmf') if hasattr(rv, 'logpmf') else logprob

    assert logprob is not None, 'no log-pdf for stat : {:s}'.format(name)
    
    ll = lambda x, *pars: np.sum(logprob(x, *pars))
    
    return ll
        
    
def stat_argument_names(name : str):
    """
    
    return the list of arguments names for a given name of an stat object

    Parameters
    ----------
    name : str
        Name of the stats-object, i.e 'gaus', 'exp+gaus'

    Returns
    -------
    argnames: tuple(str)
        List with the names of the arguments
        
    """
    
    if (name.find('+')>0):
        names = name.split('+')
        argnames = []
        for i, name in enumerate(names):
            vnames    = [r'n', ] +list(stat_argument_names(name))
            argnames += [v + '$_'+str(i)+'$' for v in vnames]
        return argnames
    
    if (name in efit_argnames.keys()):
        return efit_argnames[name]
    
    if (hasattr(stats, name)):
        rv       = getattr(stats, name)
        argnames = ['loc', 'scale'] if rv.numargs <= 0 else rv.shapes.split(',')
        return tuple(argnames)
    
    assert False, 'no arguments names for stat : {:s}'.format(name)
    
    return None    


#
#--- Composite PDF
#

class CompositePDF:
    """
    
    Stats-class with a composite of PDFs,
    
    Implementss: rvs, pdf, logpdf, loglike methods
    
    """
    
    def __init__(self, probas = tuple, nargs = None):
        """
        
        Constructor of a Composite PDF

        Parameters
        ----------
        probas : tuple(str) or tuple(pdf)
            if tuple(str) , list of the names of the pdfs,
            if tuple(pdfs), the list of pdfs objects
        nargs :  tuple(int)
            list of the number of arguments for each pdf.
            required when a tuple(pdf) is probided as first argument 
            optional when a tuple(str) is probided as first argument
         
        """

        if (type(probas[0]) == str):
            names  = tuple(probas)
            probas = [stat(name)                     for name in names]
            nargs  = [len(stat_argument_names(name)) for name in names]
            return self.__init__(probas, nargs)

        assert nargs is not None, 'Please enter the list with the number of arguments of the pdfs' 
        assert len(probas) == len(nargs), 'the probabilities and number of arguments input must be of same size'
        
        self.stats    = probas
        nargs         = [ni+1 for ni in nargs]
        nstats        = len(probas)
        
        ntot          = np.sum(nargs)
        self.mask0    = np.zeros(ntot, bool)
        self.masks    = [np.zeros(ntot, bool) for i in range(nstats)]
        for i, ni in enumerate(nargs):
            n0 = int(np.sum(nargs[:i]))
            self.mask0[n0] = True 
            self.masks[i][n0 + 1: n0 + ni] = True
        

    def rvs(self, *pars, size: int = 1):
        """
        
        generate size random data

        Parameters
        ----------
        *pars : parameters of the pdf
        size  : int, optional.
            number of random data
            As the PDF is composed, 
            in this case, size = 1 generate already a poisson number of events for each
            internal pdf, based on its size.

        Returns
        -------
        ys : np.array
            if size = 1, x-values generated with the composite PDF

        """
                    
        pars = np.array(pars)
        ys = []
        for i in range(size):
            ns = [stats.poisson.rvs(ni) for ni in pars[self.mask0]]
            xs = [istat.rvs(*pars[imask], size = int(ni)) for istat, imask, ni 
                  in zip(self.stats, self.masks, ns)]
            xs = np.concatenate(xs)
            ys.append(xs)
        ys = ys[0] if size == 1 else ys
        return ys
    
    
    def pdf(self, x, *pars):
        """
        
        probability density function
        
        Parameters
        ----------
        x     : number or np.array
        *pars : pdf parameters

        Returns
        -------
        y    : pdf(x | pars)

        """
        pars = np.array(pars)
        ns   = pars[self.mask0]
        ws   = ns/np.sum(ns)
        ys   = [wi * istat.pdf(x, *pars[imask]) for wi, istat, imask 
                in zip(ws, self.stats, self.masks)]
        y    = reduce(operator.add, ys)
        return y
        
    
    def loglike(self, x, *pars):
        """
                
        Extended Log-Likelihood

        Parameters
        ----------
        x     : np.array
        *pars : parameters of the pdf

        Returns
        -------
        ll    : float, loglike(x | pars)

        """
        #pars = self.pars if len(pars) == 0 else np.array(pars)
        pars = np.array(pars)
        y    = self.pdf(x, *pars)
        ll   = np.sum(np.log(y))
        n    = len(x)
        ni   = np.sum(pars[self.mask0])
        exll = stats.poisson.logpmf(int(n), ni)
        return ll + exll
    
        
    # def guess(self, x):
    #     pars = np.zeros(self.nargs)
    #     pars[self.mask0] = len(x)/self.nargs
    #     for ifun, imask in zip(self.funs, self.masks):
    #         assert ifun._fit is not None, 'No possible to estimate the parameters guess'
    #         pars[imask] = ifun._fit(x)
    #     return pars
        

#
#---- Internal Functions
#

def _stat_guess(name : str):
    
    rv    = stat(name)
    guess = getattr(rv, 'fit') if hasattr(rv, 'fit') else None
    
    assert guess is not None, 'Please provide a initial parameter guess'
    
    return guess


class exp:
    
    shapes = ('tau',)
            
    def rvs(tau, size = 1):
        return stats.expon.rvs(scale = tau, size = size)
    
    def pdf(x, tau):
        return stats.expon.pdf(x, scale = tau)
    
    def logpdf(x, tau):
        return stats.expon.logpdf(x, scale = tau)
    
    def fit(x):
        return np.mean(x)

