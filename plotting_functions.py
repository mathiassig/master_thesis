import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const
from scipy import interpolate
####################################################################
"Define a plotting function for log(nu)-log(nuF_nu) plot"
def plot_lognu_logF(x,y,linestyle,linewidth,color,label,xlabel,ylabel,
                    ylims=None,xlims=None,multiplot=False,title=None,
                    errorlist=None,filename=None,showfig=False,grids=False,shaded_region=None,tight_layout=False,outside_legend=False,figuresize=None):
    if figuresize:
        fig, ax = plt.subplots(figsize = figuresize, dpi=300)
    else:
        fig, ax = plt.subplots(figsize = (10,8), dpi=300)
    if multiplot:
        for i in range(len(label)):
            def plot(ax):
                ax.loglog(
                    x[i], y[i],
                    marker=None, linestyle=linestyle[i], linewidth=linewidth[i], color=color[i],
                    label=label[i],
                )

            plot(ax)

            ax.set_xlabel(f'{xlabel}, ' + str(x[0].unit),fontsize=20)
            ax.set_ylabel(f'{ylabel}, ' + str(y[0].unit),fontsize=20)
    else:
        def plot(ax):
            ax.loglog(
                x, y,
                marker=None, linestyle=linestyle, linewidth=linewidth, color=color,
                label=label,
            )

        plot(ax)
        ax.set_xlabel(f'{xlabel}, ' + str(x.unit),fontsize=20)
        ax.set_ylabel(f'{ylabel}, ' + str(y.unit),fontsize=20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if ylims:
        plt.ylim(bottom = ylims[0],top = ylims[1])
    if xlims:
        plt.xlim(left = xlims[0],right = xlims[1])
    if title:
        plt.title(title,fontsize=20)
    if errorlist:
        for error in errorlist:
            plt.errorbar(error[0],error[1],xerr=error[2],yerr=error[3],marker = error[4],
                         linestyle='',label = error[5],elinewidth=2)
    if shaded_region:
        ax.fill_between(shaded_region[0], shaded_region[1], y2=shaded_region[2], color = shaded_region[3],label = shaded_region[4])
    if outside_legend:
        plt.legend(fontsize=20,bbox_to_anchor = [1.01, 0.5], loc = 'center left')
    else:
        plt.legend()
    if grids:
        plt.grid()
    if tight_layout:
        plt.tight_layout()
    if filename:
        plt.savefig(filename)
        print(f'Figure saved as {filename}')
    if showfig:
        plt.show()

# function from agnprocesses spectra.py
def summ_spectra(e1, s1, e2, s2, nbin=100,
                 avoid_bad=True):
    """
    e1 is the energy numpy array of the 1st spectrum (SED) or array-like
    s1 is the intensity, SED and so on (1st spectrum), with the same size as e2
    e2 is the energy numpy array of the 2nd spectrum (SED)
    s2 is the intensity, SED and so on (2nd spectrum), with the same size as e1
    nbin is the number of bins in the final spectrum (SED)

    If e1 is an astropy Quantity, e2 should be an astopy Quantity.
    If s1 is an astropy Quantity, s2 should be an astopy Quantity.

    Final energy and spectrum (SED) has units of e1 and s1 correspondingly.
    """
    x_u = 1
    y_u = 1
    try:
        if e1.shape[0] != s1.shape[0]:
            raise ValueError("sizes of e1 and s1 must be equal!")
        if e2.shape[0] != s2.shape[0]:
            raise ValueError("sizes of e2 and s2 must be equal!")
    except AttributeError:
        raise AttributeError(
            "e1, s1, e2, s2 must be numpy arrays or array-like!")

    try:
        e2 = e2.to(e1.unit)
        e2 = e2.value
    except AttributeError:
        pass

    try:
        s2 = s2.to(s1.unit)
        s2 = s2.value
    except AttributeError:
        pass

    try:
        x_u = e1.unit
        e1 = e1.value
    except AttributeError:
        pass

    try:
        y_u = s1.unit
        s1 = s1.value
    except AttributeError:
        pass
    if avoid_bad:
        bad1 = np.logical_or(
            (s1 <= 0),
            np.isnan(s1)
        )
        s1[bad1] = 1.0e-40
        bad2 = np.logical_or(
            (s2 <= 0),
            np.isnan(s2)
        )
        s2[bad2] = 1.0e-40
    logx1 = np.log10(e1)
    logy1 = np.log10(s1)
    f1 = interpolate.interp1d(
        logx1, logy1, kind="linear", bounds_error=False, fill_value=(-40, -40)
    )
    logx2 = np.log10(e2)
    logy2 = np.log10(s2)
    f2 = interpolate.interp1d(
        logx2, logy2, kind="linear", bounds_error=False, fill_value=(-40, -40)
    )
    emin = np.min((np.min(e1), np.min(e2)))
    emax = np.max((np.max(e1), np.max(e2)))
    e = np.logspace(np.log10(emin), np.log10(emax), nbin)
    x = np.log10(e)
    s = (10.0 ** (f1(x)) + 10.0 ** (f2(x))) * y_u  # new summed spectrum (SED)
    e = e * x_u  # new energy
    return (e, s)

## for the errorbar calculations ##
def fill_dictionaries(data,dictionary):
    for element in data:
        dictionary[element[3]].append([float(element[0]),float(element[1])]) # x and y coords
    for i in range(len(dictionary)):
        dictionary[data[i,3]] = np.array(dictionary[data[i,3]])
    return
def nonelist(length):
    non = []
    for i in range(length):
        non.append(None)
    return non
