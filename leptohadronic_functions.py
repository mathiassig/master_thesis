import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo
from scipy.integrate import quad
from numba import jit
#################################################################################
# function from agnprocesses cosmology.py
def luminosity_distance(redshift):
    return cosmo.luminosity_distance(redshift)
# function from agnprocesses relativity.py
@jit(nopython=True)
def gamma_to_beta(gamma):
    """
    Given Lorentz factor gamma return the corresponding value
    beta of the velocity divided by the speed of light.
    """
    return(1.0 - gamma**(-2))**0.5
# function from agnprocesses relativity.py
@jit(nopython=True)
def doppler_factor(gamma, theta):
    """
    Given Lorentz factor gamma and viewing angle theta return the corresponding value
    delta of the Doppler factor.
    """
    return((gamma * (1.0 - gamma_to_beta(gamma) * np.cos(theta)))**(-1))
######################## External photon field ##################################

# The following is copied from https://am3.readthedocs.io/en/latest/examples/detailed_example.html
# Based on the article by Rodrigues et al. (2023): https://arxiv.org/abs/2307.13024
#"Below we define arrays with the spectral shapes of these different components,
# normalize them to the disk luminosity including the respective covering factors,
# boost them into the jet rest frame, add them up into a single array,
# and finally inject this in the simulation as external photons."
def PlanckDistribution(earr, temperature, lum):
    '''
    Thermal Distribution (unnormalized)

    return: E^2dN/dE [a.u.]
    par earr (array): photon energy [eV]
    par temperature [K]
    par lum: total luminosity [erg/s]
    '''
    lgr = earr / temperature / const.k_B.to(u.eV/u.K)
    exparr = np.exp(lgr) - 1
    ednde = earr ** 4 / exparr
    integ = np.trapz(ednde / earr, earr)
    return ednde * lum / integ


def Schwarzschild(m_bh):
    '''Schwarzschild radius [cm]

    m_bh: black hole mass [m_solar]
    '''
    rad = (2 * const.G * m_bh * 1.989e30*u.kg
           / const.c ** 2)
    return rad.to(u.cm)


def DiskTemperature(rad, lumdisk, m_bh, eta=0.08):
    '''Radius-dependent disk temperature [K]
    '''
    sb = const.sigma_sb.to(u.erg/u.s/
                           u.K**4/u.cm**2)
    rsch = Schwarzschild(m_bh)

    term1 = 3 * rsch * lumdisk / (16 * np.pi * eta
                                  * sb * rad ** 3)
    term2 = 1 - (3 * rsch / rad) ** 0.5
    return (term1 * term2) ** 0.25


def ShakuraFlux(earr, lumdisk, m_bh, thetaobs=3.0, eta=0.08):
    '''Disk spectral flux in observer's frame [erg/cm2/s]

    earr (array): photon energies [eV]
    lumdisk: [erg/s]
    m_bh: black hole mass / m_solar
    thetaobs: angle btw. LOS and disk rotation axis (deg)
    '''
    kB = const.k_B.to(u.eV/u.K)
    c0 = const.c.cgs
    hplanck = const.h.to(u.eV*u.s)
#     dlum = cosmo.luminosity_distance(z).to(u.cm)
    rsch = Schwarzschild(m_bh)
    rin = 3 * rsch
    rout = 300 * rsch
    radarr = np.linspace(rin, rout, 50)

    frac = (4 * np.pi) ** 2 * hplanck * np.cos(thetaobs*np.pi/180) / c0 ** 2
    nuarr = earr / hplanck       # convert x-axis to obs frame
    en2d = earr[:,np.newaxis]    # convert x-axis to obs frame
    rad2d = radarr[np.newaxis,:]
    temp2d = DiskTemperature(rad2d,lumdisk,m_bh,eta) # K

    integrand = rad2d / (np.exp(en2d/temp2d/kB) - 1)
    integral = np.trapz(integrand, radarr, axis=1)
    Fnu = (nuarr ** 3 * frac * integral).to(u.erg)
    return nuarr * Fnu

def BroadLine(earr, center, width, lum):
    '''
    Broad line spectrum, normalized to `lum`

    return: E^2dN/dE [a.u.]
    par earr (array): photon energies (eV)
    par center: line energy [eV]
    par width: line width (eV)
    par lum: line luminosity [erg/s]
    '''
    ednde = np.exp(-0.5
                   * (earr - center) ** 2
                   / width ** 2
                  )
#     ednde[ednde < 1e-100] = 0.
    integ = np.trapz(ednde/earr, earr)
    ednde *= lum / integ
    return ednde

def BroadLineBB(earr, temp, lum):
    '''
    Broad line spectrum as a black body, normalized to `lum`

    return: E^2dN/dE [a.u.]
    par earr (array): photon energies (eV)
    par temp: temperature (eV)
    par lum: line luminosity [erg/s]
    '''
    ednde = (earr.value**3)/(np.exp(earr/temp)-1)
#     ednde[ednde < 1e-100] = 0.
    integ = np.trapz(ednde/earr, earr)
    ednde *= lum / integ
    return ednde

def get_BLR_density_scaling(R_zone, R_diss, lorentz):
    '''Scaling of the photon density seen in the jet frame
    with the dissipation radius, according to Eq. 20 of
    Ghisellini+Tavecchio 0902.0793
    '''

    def scaling_for_large_R_diss(R_diss, R_zone, lorentz):
        beta = (1 - 1. / lorentz ** 2) ** 0.5
        mu1 = (1 + (R_zone / R_diss) ** 2) ** -.5
        mu2 = (1 - (R_zone / R_diss) ** 2) ** .5
        f_mu = (2 * (1 - beta * mu1) ** 3
                - (1 - beta * mu2) ** 3
                - (1 - beta) ** 3)
        return f_mu / 3. / beta

    f0 = 17. / 12

    if R_diss <= R_zone:
        scaling = f0

    elif R_diss >= 3 * R_zone:
        scaling = scaling_for_large_R_diss(
            R_diss,
            R_zone,
            lorentz)

    elif R_zone < R_diss < 3 * R_zone:
        # Power-law interpolation
        f_3R = scaling_for_large_R_diss(
            3 * R_zone,
            R_zone,
            lorentz
        )
        scaling = 10 **(
            (np.log10(f_3R) - np.log10(f0))
            / (np.log10(3 * R_zone) - np.log10(R_zone))
            * (np.log10(R_diss) - np.log10(R_zone))
        )

    return scaling

def tangential_angle(R_BLR, R_diss):
    '''Calculate the characteristic angle of the radiaiton,
        which is the tangential angle. This is where the dominant
        contribution comes from because it has the highest doppler
        boost, as well as for geometric reasons.
    '''
    csi = np.arcsin(R_BLR/R_diss)
    return csi

def calc_doppler(lorentz, R_BLR, R_diss):
    '''Calculate relative Doppler factor between blob and BLR.
        The blob has bulk factor `lorentz` and distance to the black hole
        given by `R_diss` [cm]. The BLR has radius `R_BLR` [cm].
    '''
    if R_diss <= R_BLR:
        return lorentz

    csi = tangential_angle(R_BLR, R_diss)
    beta = (1 - 1. / lorentz ** 2) ** 0.5
    doppler = lorentz * (1 - beta * np.cos(csi))
    return doppler

def convert_lum_to_density_in_jet(R_diss, lorentz, R_BLR):
        ''' Convert external field luminosity in the rest frame of the
        black hole in [erg/s] into energy density in the comoving frame
        of the jet blob in [GeV / cm^3]. R_BLR can represent the BLR radius
        or the dust torus radius.

        return: conversion factor [GeV erg^-1 s cm^-3]
        '''

        #doppler_fact = calc_doppler(lorentz, R_BLR, R_diss)

        f1 = (lorentz ** 2 # it was doppler_factor in the original version, however that is wrong
                / (4.
                    * np.pi
                    * R_BLR ** 2
                    * const.c.cgs.value)
                * u.erg.to('GeV'))

        f2 = get_BLR_density_scaling(R_BLR, R_diss, lorentz)
        factor =  f1 * f2

        return factor

def Corona(earr,corona_minimum_energy,corona_maximum_energy,corona_spectral_index,corona_luminosity,luminosity_distance,disky,disk_e):
    # See Ghisellini & Tavecchio (2009)
    # valid for 0.1-10 keV, about 10¹⁶ to 10¹⁸ Hz
    # take in energy array with values in eVs
    def Lx_e_core(x): #  spectral luminosity, corona
        return x**(-corona_spectral_index)*np.exp(-(x/corona_maximum_energy).value) # K depends on Ld
    def Lx_e(earr,corona_minimum_energy,corona_maximum_energy,corona_luminosity):
        I = quad(Lx_e_core,corona_minimum_energy.value, corona_maximum_energy.value)
        return corona_luminosity/I[0]*Lx_e_core(earr.value)

    lum = Lx_e(earr,corona_minimum_energy,corona_maximum_energy,corona_luminosity)
    lum = np.where(lum == np.inf,0,lum) # remove stupid divide by zero error
    sed = earr.value*lum/(4*np.pi*luminosity_distance**2)
    sed = sed.to(u.erg/(u.s*u.cm**2))
    # a = np.where(disky.value < np.max(sed.value))[0]
    # k = 0
    # for ind in a:
    #     if ind>len(disky)/2:
    #         k = ind
    #         break
    # join_e = disk_e[k-1]
    sed = np.where(earr<corona_minimum_energy*0.1,0,sed)
    return sed
