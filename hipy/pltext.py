"""

  Customize matplotlib and histogram plotting
  
  Plotting to fitted histograms and profiles

"""


import numpy  as np
import pandas as pd


import hipy.utils  as ut
import hipy.cfit   as cfit
import hipy.histos as histos
#from   dataclasses import dataclass

import matplotlib.pyplot as plt
from   mpl_toolkits.axes_grid1 import make_axes_locatable
from   cycler import cycler

# cmaps:
# magma, inferno, Red, Greys, Blues, spring, summer, autumn, winter,
# cool, Wistia, hot, jet

# colorbar
# cbar = plt.colorbar(heatmap)
# cbar.ax.set_yticklabels(['0','1','2','>3'])
# cbar.set_label('# of contacts', rotation=270)
#


def style():
    """ my mathplot style
    """

    plt.rcParams['axes.prop_cycle'] = cycler(color='kbgrcmy')
    plt.style.context('seaborn-colorblind')
    return


def locate_text(comment : str, 
                x : float = 0.05,
                y: float = 0.7,
                **kargs):
    props = dict(boxstyle='square', facecolor='white', alpha= 0.5)
    plt.gca().text(x, y, comment, transform = plt.gca().transAxes, bbox = props, **kargs)
    return


def canvas(ns : int, 
           ny : int = 2,
           height : float = 5.,
           width : float = 6.) -> callable:
    """

    Create a canvas with ns subplots and ny-columns.
    Return a function to move to next subplot in the canvas


    Parameters
    ----------
    ns     : int, total number of subplots
    ny     :  int, number of columns. Default 2.
    height : float, optional, height of the subplot. Default is 5.
    width  : float, optional, width of the subplot. Default is 6.

    Returns
    -------
    subplot: function (i : int, str), to locate a subplot in posiiton i,
             if str = '3d' a 3D subplot is created
    """

    nx  = int(ns / ny + ns % ny)
    plt.figure(figsize = (width * ny, height * nx))

    def subplot(iplot : int , dim: str = '2d'):
        """
        Set the current plot in the iplot position of the canvas

        Parameters
        ----------
        iplot : int, iplot number of the canvas.
        dim    : str, optional.The default is '2d'. if '3d' a 3D subplot is created
        """
        assert iplot <= nx * ny, 'Not valid number of subplot'
        plt.subplot(nx, ny, iplot)
        if (dim == '3d'):
            nn = nx * 100 +ny *10 + iplot
            plt.gcf().add_subplot(nn, projection = dim)
        return

    return subplot


def split_plot(ax = None, size = 20):
    
    ax = plt.gca() if ax is None else ax
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size = str(20)+'%', pad = 0)
    ax.figure.add_axes(ax2)
    
    return ax2
    
#
# string conversions
#----------------

formate     = '6.2f'

def label_stats(size    : float,
                mean    : float,
                std     : float,
                formate : str = formate) -> str:
    s  = 'entries '+str(size)+'\n'
    s += (('mean {0:'+formate+'}').format(mean))+'\n'
    s += (('std  {0:'+formate+'}').format(std))
    return s


def label_parameter(name    : str,
                    value   : float,
                    error   : float = None,
                    formate : str = formate) -> str:
    s = name + ' = '
    s += (('{0:'+formate+'}').format(value))
    if (error != None):
        s += (r'$\pm$ {0:'+formate+'}').format(error)
    return s


def label_parameters(pars     : np.array, 
                     upars    : np.array = None,
                     parnames : tuple = None,
                     formate  : str = formate) -> str:
    
    label = ''
    upars = len(pars)* [None,] if upars is None else upars
    for pname, par, upar in zip(parnames, pars, upars):
        label += label_parameter(pname, par, upar, formate) + '\n'
    return label


#   HISTOGRAMS
#----------------------


def hist(x        : np.array, 
         bins     : int, 
         range    : tuple = None,
         stats    : bool = True,
         formate  : str = '6.3f',
         xylabels : tuple = None,
         grid     : bool = True,
         ylog     : bool= False,
         **kargs):
    """

    Decorate histogram

    Parameters
    ----------
    x        : np.array.
    bins     : int. number of bins or array of bins.
    range    : tuple. Histogram range. Default is None.
    stats    : bool. Put the statistic on the legend. The default is True.
    formate  : str. Formate of the statistic in plot. Default '6.3f'
    xylabels : tuple(str), optional. Write the labels of the histogram (xlabel, ylabel). The default is None.
    grid     : bool, optional. Draw the grid. The default is True.
    ylog     : bool, optional. Y-scale log. The dafult is False.
    **kargs  : dict. Other key arguments of plt.hist

    Returns
    -------
    c       : tuple, return of plt.hist
    
    """

    if (stats):
        x    = np.array(x)
        xp   = x[ut.in_range(x, range)]
        ss   = label_stats(*ut.stats(xp), formate) 

        if ('label' in kargs.keys()):
            kargs['label'] += '\n' + ss
        else:
            kargs['label'] = ss

    if (not ('histtype' in kargs.keys())):
        kargs['histtype'] = 'step'

    c = plt.hist(x, bins, range, **kargs)


    if (xylabels is not None):

        if (type(xylabels) == str):
            plt.xlabel(xylabels)

        if (type(xylabels) == tuple):
            xlabel, ylabel = xylabels[0], xylabels[1]
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)


    if ('label' in kargs.keys()):
        plt.legend()

    if (grid): plt.grid(True)

    if (ylog): plt.yscale('log')

    return c

#   Histogram Fit
#--------------------

def hfit(x         : np.array, 
         bins      : int,
         fun       : str, 
         range     : tuple = None, 
         p0        : np.array = None, 
         parnames  : tuple = None,
         formate   : str = formate,
         mode      : str = 'hist', 
         residuals : bool = False,
         fitopts   : dict = {},
         funopts   : dict = {},
         **kargs
          ):
    """
    
    Plot a fit a histogram to a function via name

    Parameters
    ----------
    x         : np.array
    bins      : int. Number of bins
    fun       : str. Name of the function. i.e 'gaus', 'exp' or 'gaus+exp'
    range     : tuple, optional. x-range
    p0        : np.array, optional. Initial parameter guess. The Default is None.
    parnames  : tuple, optional. Names of the parameters. The default is None.
    formate   : str, optional. Formate of the parameters in the legend. The default is formate.
    mode      : str, optional. Plot mode: 'hist', 'plot'. The default is 'hist'.
    residuals : bool, optional subplot with the residuals. The default is False.
    fitopts   : dict, optional. Key options for curve_fit. The default is {}.
    funopts   : dict, optional. Key options for plotting the function. The default is {}.
    **kargs   : 

    Returns
    -------
    contents : np.array. Contents of the histogram
    edges    : np.array. Edges of the histogram
    pars     : np.array. Parameters of the fit
    upars    : np.array. Uncertainties of the parameters
    ffun     : callable. The fitted function, y = fun(x)
    
    """
    
    yc, edges = np.histogram(x, bins, range)
    xc        = 0.5 * (edges[1:] + edges[:-1])
    ye        = np.maximum(2.4, np.sqrt(yc))
    
    pars, epars, ffun = cfit.curve_fit(xc, yc, fun, p0 = p0,
                                             sigma = ye, **fitopts)
    
    result = yc, edges, ye, pars, epars, ffun
    
    parnames = cfit.curve_argument_names(fun) if parnames is None else parnames
    label    = label_parameters(pars, epars, parnames, formate = formate)

    hfun(edges, yc, ffun, ye, label = label, mode = mode,
         residuals = residuals, funopts = funopts, **kargs)
        
    return result



def hfun(x         : np.array,
         y         : np.array,
         fun       : callable,
         ye        : np.array = None,
         mode      : str = 'hist',
         residuals : bool = False,
         funopts   : dict = {},
         **kargs
         ):
    """
    
    Plot the (x, y) data and overlaid the function y = fun(x)

    Parameters
    ----------
    x         : np.array
    y         : np.array
    fun       : callable
    ye        : np.array, optional. T errors. The default is None.
    mode      : str, optional. Mode of plotting, 'hist' or 'plot'. The default is 'hist'.
    residuals : bool, optional. Subplot with the residuals. The default is False.
    funopts   : dict, optional. Key parameters to draw the function. The default is {}.
    **kargs   : key parameters to draw the (x, y) plot

    """

    
    ye = ut.yerrors(y) if ye is None else ye    
    
    xc = x if len(x) == len(y) else ut.centers(x)
    
    assert (len(xc) == len(y)), 'Required sime size for x and y to plot'
        
    if (mode == 'plot'):
        sel = y > 0
        kopts = dict(kargs)
        if 'marker' not in kopts: kopts['marker'] = 'o'
        if 'ls'     not in kopts: kopts['ls']     = ''
        plt.errorbar(xc[sel], y[sel], ye[sel],  **kopts)
        plt.grid();
    else:
        kopts = dict(kargs)
        if ('stats' not in kopts.keys()): kopts['stats'] = False
        hist(xc, x, weights = y, **kopts)
    if 'label' in kargs.keys(): plt.legend()
   
    plt.plot(xc, fun(xc), **funopts);
    
    if (residuals):
        split_plot()       
        hresiduals(xc, y, fun, ye, **funopts)

    return


def hresiduals(x       : np.array,
               y       : np.array,
               fun     : callable,
               ye      : np.array = None,
               formate : str = formate,
               **kargs   
               ):
    """
    
    Plot the residuals res = (y - fun(x))/yerr

    Parameters
    ----------
    x       : np.array
    y       : np.array
    fun     : callable
    ye      : np.array, optional. y errors. The default is None.
    formate : str, optional. Formate fot the legend. The default is formate.
    **kargs : 

    """

    ye = ut.yerrors(y) if ye is None else ye    
    
    xc = x if len(x) == len(y) else ut.centers(x)
    
    assert (len(xc) == len(y)), 'Required sime size for x and y to plot'

    # # subplotting of the residuals    
    # ax = plt.gca()
    # divider = make_axes_locatable(ax)
    # ax2 = divider.append_axes("bottom", size = '20%', pad = 0)
    # ax.figure.add_axes(ax2)

    res  = (y - fun(xc))/ye
    chi2 = np.sum(res*res)/len(res) 

    label = label_parameter(r'$\chi^2$/ndf', chi2, formate = formate)
    width = np.min(np.diff(x))
    if 'label' not in kargs.keys():
        kargs['label'] = label
    else:
        kargs['label'] += '\n ' + label
    plt.bar(xc, res, width = width, **kargs);
    plt.grid()
    if 'label' in kargs.keys(): plt.legend()

    return


#     PROFILES
#----------------


def hprofile(x      : np.array, 
             y      : np.array, 
             bins   : int, 
             xrange : tuple = None,
             std    : bool = False, 
             percentile : bool = False,
             **kargs):
    """
    
    Plot the profile of y vs x. 
    If std the standard deviation are plotted as errors.

    Parameters
    ----------
    x       : np.array
    y       : np.array
    bins    : int, number of bins, or np.array with the bins
    xrange  : tuple, optional. The default is None.
    std     : bool, standard deviation or mean error. The default is False.
    percentile : bool, equal number of counts bins. Defaults is False
    **kargs : dict, key arguments for plot.

    Returns
    -------
    counts : np.array, counts in x-bins
    xmean  : np.array, x-mean of the x-bins
    xstd   : np.array, x-std of the x-bins
    ymean  : np.array, y-mean in x-bins
    ystd   : np.array, y-std  in x-bins
    """
    
    counts, xmean, xstd, ymean, ystd =  \
        histos.hprofile(x, y, bins, xrange, percentile = percentile)
    
    yerr = ystd if std else ystd/np.sqrt(counts)
    xerr = xstd if std else xstd/np.sqrt(counts)
    plt.errorbar(xmean, ymean, yerr, xerr, **kargs)
    
    return counts, xmean, xstd, ymean, ystd


def hfitprofile(x         : np.array, 
                y         : np.array,
                bins      : int,
                fun       : str, 
                xrange    : tuple = None, 
                std       : bool = False,
                percentile : bool = False,
                p0        : np.array = None, 
                parnames  : tuple = None,
                formate   : str = formate,
                mode      : str = 'plot', 
                residuals : bool = False,
                fitopts   : dict = {},
                funopts   : dict = {},
                **kargs
                ):
    """
    
    Plot the profile and overlaid the fit to a curve by name

    Parameters
    ----------
    x         : np.array.
    y         : np.array.
    bins      : int. number of bins
    fun       : str. name of the function. i.e 'poly.1'
    xrange    : tuple, optional. x-range. The default is None.
    std       : bool, optional. Use as yerrors the std. The default is False.
    percentile : bool, equal number of counts bins. Defaults is False
    p0        : np.array, optional. Initial parameters to start the fit.The default is None.
    parnames  : tuple, optional. Names of the parameters. The default is None.
    formate   : str, optional. Formate for the parameters on the legend. The default is formate.
    mode      : str, optional. Plot mode: 'hist', 'plot'. The default is 'plot'.
    residuals : bool, optional. Add subplot with the residuals. The default is False.
    fitopts   : dict, optional. Key options to cuve_fit. The default is {}.
    funopts   : dict, optional. Key options to plot the funtion. The default is {}.
    **kargs   : dict, optional. Key options to plot the (x, y) data.

    Returns
    -------
    xmean  : np.array, x-mean of the x-bins.
    ymean  : np.array, x-mean of the x-bins.
    yerr   : np.array. Errors of the y values.
    pars   : np.array. Parameters of the fit.
    upars  : np.array. Errors of the parameters.
    ffun   : callable. Function y = f(x) of the fit.

    """
    
    counts, xmean, xstd, ymean, ystd = \
        histos.hprofile(x, y, bins, xrange, percentile = percentile)
     
    
    yerr = ystd/np.sqrt(counts)
    pars, upars, ffun \
        = cfit.curve_fit(xmean, ymean, fun,  p0 = p0, sigma = yerr, **fitopts)

    parnames = cfit.curve_argument_names(fun) if parnames is None else parnames
    label    = label_parameters(pars, upars, parnames, formate = formate)

    yerr = ystd if std else yerr
    hfun(xmean, ymean, ffun, yerr, label = label,
         mode = mode, residuals = residuals,
         funopts = funopts, **kargs)
    
    return xmean, ymean, yerr, pars, upars, ffun


#    DATA FRAME
#-------------------------

def df_inspect(dfs: pd.DataFrame,
               labels: tuple = None,
               bins: int = 100, 
               ranges: dict = {},
               dfnames = None,
               ncolumns: int = 2,
               **kargs):
    """
    
    Histogram the columns of a data-frame

    Parameters
    ----------
    dfs      : pd.DataFrame or tuple(pd.DataFrame).
    labels   : tuple(str), names of the columns to histogram. The default is None.
    bins     : int, number of bins of the histograms. The default is 100.
    ranges   : dict(name: range). Ranges of the histograms.The default is {}.
    dfnames  : tuple(str), list of the DF names. The default is None.
    ncolumns : int, optional. number of columns in the canvas. The default is 2.
    kargs    : dict, key arguments for hist
    
    """
    
    dfs = (dfs,) if isinstance(dfs, pd.DataFrame) else dfs
    dfnames = ['df'+str(i) for i in range(len(dfs))] if dfnames is None else dfnames
    assert (len(dfs) == len(dfnames)), 'same number of DFs and names required'
    
    labels = tuple(dfs[0].columns) if labels is None else labels
    subplot = canvas(len(labels), ncolumns)
    for i, label in enumerate(labels):
        subplot(i + 1)
        xrange = None if label not in ranges.keys() else ranges[label]
        for df, dfname in zip(dfs, dfnames):
            values = ut.remove_nan(df[label].values)
            hist(values, bins, range = xrange, label = dfname, **kargs)
            plt.xlabel(label);
        plt.legend()
    plt.tight_layout()
    return


# def df_corrmatrix(xdf, xlabels):
#     """ plot the correlation matrix of the selected labels from the dataframe
#     inputs:
#         xdf     : DataFrame
#         xlabels : tuple(str) list of the labels of the DF to compute the correlation matrix
#     """
#     _df  = xdf[xlabels]
#     corr = _df.corr()
#     fig = plt.figure(figsize=(12, 10))
#     #corr.style.background_gradient(cmap='Greys').set_precision(2)
#     plt.matshow(abs(corr), fignum = fig.number, cmap = 'Greys')
#     plt.xticks(range(_df.shape[1]), _df.columns, fontsize=14, rotation=45)
#     plt.yticks(range(_df.shape[1]), _df.columns, fontsize=14)
#     cb = plt.colorbar()
#     cb.ax.tick_params(labelsize=14)
#     return

#
# def df_corrprofile(df, name, labels, switch = False, **kargs):
#     """ plot the scatter and profile plot between the name-variable
#     of the df, DataFrame, vs each variable in labels list
#     inpùts:
#         df    : DataFrame
#         name  : str, name of the variable for the x-axis profile
#         labels: list(str), names of the variable sfor the y-axis profile
#         swicth: bool, False. Switch x-variable and y-variable
#     """
#     sargs = dict(kargs)
#     if 'alpha' not in sargs.keys(): sargs['alpha'] = 0.1
#     if 'c'     not in sargs.keys(): sargs['x']     = 'grey'
#
#
#     subplot = canvas(len(labels), len(labels))
#     for i, label in enumerate(labels):
#         subplot(i + 1)
#         xlabel, ylabel = (name, label) if switch is False else (label, name)
#         kargs['alpha'] = 0.1    if 'alpha' not in sargs.keys() else sargs['alpha']
#         kargs['c']     = 'grey' if 'c'     not in sargs.keys() else sargs['c']
#         plt   .scatter (df[xlabel], df[ylabel], **kargs)
#         kargs['alpha'] = 1.
#         kargs['lw']    = 1.5 if 'lw'    not in sargs.keys() else sargs['lw']
#         kargs['c']     = 'black'
#         hprofile(df[xlabel], df[ylabel], **kargs)
#         plt.xlabel(xlabel, fontsize = 12); plt.ylabel(ylabel, fontsize = 12);
#     plt.tight_layout()
#     return
