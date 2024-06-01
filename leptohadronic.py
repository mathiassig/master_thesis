# This program makes SEDs,
# based on the input parameters given in
# 2011MNRAS.411..901G
# URL: https://ui.adsabs.harvard.edu/abs/2011MNRAS.411..901G/abstract
# below is a lepto-hadronic model with the following radiation processes:
# synchrotron, SSC, inverse compton; in addition to thermal radiation from
# the dust torus (IR), accretion disk (visual), corona (x-ray)
################################################################################
#Import libraries for modelling AGN processes
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from astropy import units as u
from astropy import constants as const
from plotting_functions import *
from leptohadronic_functions import * # functions defined by myself + copied from online examples
import sys
sys.path.append("../../libpython/") # change to path to where AM3 is stored on your computer

################################################################################
inputs = np.loadtxt('input_parameters.txt',delimiter = "; ",dtype=str)#load the input parameters
number_of_sources = len(inputs[:,0])-1
input_4FGL = np.loadtxt('4fgl.txt',delimiter = "; ",dtype=str)
input_neutrino = np.loadtxt('neutrino_sensitivity.txt',delimiter = "; ",dtype=str)
input_proton = np.loadtxt('proton_max_energy.txt',delimiter = "\n",dtype=str) # load maximum proton energies
input_powers = np.loadtxt('particlepowers.txt',delimiter = "; ",dtype=str) # load weighting factors for electron and proton powers
file_nu = open("neutrino_luminosity.txt","w") # to store bolometric neutrino luminosity
file_nu_barplot= open("neutrino_barplot.txt","w") # to store ratio between peak neutrino flux and sensitivity
file_nu_barplot.write(f"sourcename; peak_neutrinoflux_per_icecubegen2; peak_neutrinoflux_per_arca230\n")
file_gamma = open("gamma_luminosity.txt","w")
file_gamma.write("sourcename; jet_photon_luminosity\n")

# run simulation for all 19 sources
for i in range(number_of_sources): # produces SEDs for all 19 sources in the paper
    "Define geometry, distance, magnetic field and all other input parameters"
    ## source specific input parameters ##
    sourcename = inputs[i+1,0] # name of source
    z = float(inputs[i+1,1])  # source redshift
    Rdiss = float(inputs[i+1,2])*10**15 * u.cm # distance from BH to the dissipation region
    M = float(inputs[i+1,3]) # black hole mass in solar masses
    P_i = float(input_powers[i+1,1])*float(inputs[i+1,4])*10**(45)*u.erg/u.s # power carried by the blob in the form of
                                             # bulk motion of electrons, comoving
    Ld = float(inputs[i+1,5])*10**45*u.erg/u.s # accretion disk luminosity
    b = float(inputs[i+1,6]) * u.G  # magnetic field induction
    Gamma = float(inputs[i+1,7])
    # electron spectrum
    e_br = float(inputs[i+1,8])*const.m_e.to(u.eV, u.mass_energy() # break energy, injected spectrum
                             ).value * u.eV # electron rest energy times gamma_b, observing frame
    e_max_e = float(inputs[i+1,9])*const.m_e.to(u.eV, u.mass_energy()
                                  ).value * u.eV # rest energy times gamma_max, observing frame
    gamma1 = float(inputs[i+1,10]) # powerlaw index 1, injected spectrum (s1 in table)
    gamma2 = float(inputs[i+1,11]) # powerlaw index 2, injected spectrum (s2 in table)
    e_min_e = float(inputs[i+1,12])*const.m_e.to(u.eV, u.mass_energy()).value * u.eV # rest energy*gamma_cool
    # proton spectrum
    p_br = 1.0*const.m_p.to(u.eV, u.mass_energy() # break energy, injected spectrum
                             ).value * u.eV # proton rest energy times gamma_b, observing frame
    p_max_e = float(input_proton[i])*const.m_p.to(u.eV, u.mass_energy()
                                  ).value * u.eV # rest energy times gamma_max, observing frame
    p_min_e = 1.0*const.m_p.to(u.eV, u.mass_energy()).value * u.eV # rest energy*gamma_cool
    P_i_p = float(input_powers[i+1,2])*P_i # proton power proportional to electron power
    P_index1 = 2.0 # power law index below break
    P_index2 = 2.0 # power law index above break
    ##############################################################################################

    # parameters for the whole population + derived quantities
    theta_obs_deg = 3.0 # viewing angle in degrees
    doppler = doppler_factor(Gamma,theta_obs_deg*np.pi/180.0)
    r_b = Rdiss/(Gamma)# blob radius
    d_l = luminosity_distance(z).to(u.cm)

    escape_timescale = (r_b / const.c).to(u.s)

    # parameters for disk luminosity spectrum
    # We must make some assumptions on the size and temperature of the disk.
    # The temperature will go like R**(-3/4).
    Tmax = 3.5e+4*(Ld.value*10**(-46)/(
        (M*10**(-9))**2))**0.25*u.K # maximum temperature of disk, eq. 8.11 in Ghisellini book
    Tout = Tmax*((500/3)**(-3/4)) # temperature of outer edge of accretion disk, eq. 8.5 ibid
    #and used R_in = 3*R_S, R_out = 500*R_S

    # BLR parameters
    blr_radius = 10**17*(Ld.value*10**(-45))**0.5*u.cm
    blr_covering_factor = 0.1
    # The BLR temperature is a blackbody peaking at the Lyman alpha line (multiply by Gamma to move to blob frame)
    blr_temperature = 2.47*10**15/(5.879* # use Wien's displacement law and insert Lyman alpha rest frequency
                                  10**10)*u.K # given in the rest frame of the BLR
    # IR torus parameters
    torus_radius = 2.5*10**18*(Ld.value*10**(-45))**0.5*u.cm
    torus_temperature = 3*10**13/(5.879* # use Wien's displacement law
                              10**10)*u.K # equals 510 K
    torus_covering_factor = 0.5
    # Corona parameters
    corona_minimum_energy = 0.1e+3*u.eV
    corona_maximum_energy = 150e+3*u.eV
    corona_spectral_index = 1.0
    fx = 0.3 # Lx = fx*Ld = luminosity of corona
    corona_covering_factor = 0.01*fx

    ##################################### AM3 setup ########################################

    # Once you wish to change the magnetic field strength, the AM3 module has to be imported again and a new kernel initialized.
    import am3lib
    am3 = am3lib.AM3() # initialize an AM3 object
    am3.update_energy_grid(1e-6, 1e9, 1.0e+21)
    # the setup of the simulation below is mostly copied from https://am3.readthedocs.io/en/latest/examples/simple_example.html

    # For all set functions introduced above the current settings may be retrieved replacing set with get in the function name.
    #
    # The current particle density spectra can be accessed in the form of Numpy arrays in units of cm⁻³, using am3.get_<particle>()

    ############# set switches ##############

    am3.set_verbosity_level(0) # don't print much information
    am3.set_process_parse_sed(1) # parse SED by components

    am3.set_process_escape(1) #Escape ON
    # expansion related
    am3.set_process_adiabatic_cooling(0) # adiabatic cooling OFF
    am3.set_process_expansion(0) # plasma dilution due to expansion OFF

    # synchrotron related
    am3.set_process_electron_syn(1) #electron synchrotron ON
    am3.set_process_ssa(1) # electron SSA ON
    am3.set_process_electron_syn_cooling(1) # cooling from synchrotron ON
    am3.set_process_proton_syn(1) #proton synchrotron ON
    am3.set_process_pion_syn(0) #pion synchrotron OFF
    am3.set_process_muon_syn(0) #muon synchrotron OFF

    # inverse Compton related
    am3.set_process_electron_compton(1) #electron inverse Compton ON
    am3.set_process_electron_compton_cooling(1) # cooling from Compton ON
    am3.set_process_proton_compton(1) #proton inverse Compton ON
    am3.set_process_pion_compton(0) #pion inverse compton OFF
    am3.set_process_muon_compton(0) #muon inverse compton OFF

    # secondary decay
    am3.set_process_pion_decay(1) #pions decay ON (important for neutrino production!)
    am3.set_process_muon_decay(1) #muon decay ON (iportant for neutrino production!)

    # pair production (gamma+gamma -> e- + e+)
    am3.set_process_annihilation(1) #gamma gamma annihilation ON

    # p-gamma
    am3.set_process_photopion(1) #Photo-pion production ON

    # proton proton
    am3.set_process_pp(0) # Proton proton pion production OFF

    # Bethe-Heitler
    am3.set_process_bethe_heitler(1) #Bethe Heitler Photo pair production ON

    # Initialize the kernels with the above switches
    am3.init_kernels()
    timestep = escape_timescale.value*10**(-2) # choose timestep for simulation
    am3.set_solver_time_step(timestep)
    am3.set_escape_timescale(escape_timescale.value)
    am3.set_mag_field(b.value)
    # am3.set_pp_target_proton_density(protons_per_cm3) # run if pp interactions are switched on, am3.set_process_pp(1).

    ######## injection parameters #############

    # the below function sets the dN/dE spectrum??
    am3.set_powerlaw_injection_parameters_electrons(
        (4*np.pi/3*r_b**3).value, #volume_in_cm3
        P_i.value, #injection_luminosity_in_erg_per_s
        e_min_e.value, #emin_in_eV
        e_br.value, #ebreak_in_eV
        e_max_e.value, #emax_in_eV
        gamma1, #index_below_break
        gamma2, #index_above_break
        1.0 # cutoff_steepness
        )

    am3.set_powerlaw_injection_parameters_protons(
        (4*np.pi/3*r_b**3).value,# volume_in_cm3
        P_i_p.value,# injection_luminosity_in_erg_per_s
        p_min_e.value,# pmin_in_eV
        p_br.value,# pbreak_in_eV
        p_max_e.value,# pmax_in_eV
        P_index1,# index_below_break
        P_index2,# index_above_break
        1.0 # cutoff_steepness
        )

    # The following is copied from https://am3.readthedocs.io/en/latest/examples/detailed_example.html
    # Based on the article by Rodrigues et al. (2023): https://arxiv.org/abs/2307.13024
    #"Below we define arrays with the spectral shapes of these different components,
    # normalize them to the disk luminosity including the respective covering factors,
    # boost them into the jet rest frame, add them up into a single array,
    # and finally inject this in the simulation as external photons."

    blr_doppler = calc_doppler(Gamma, blr_radius.value, Rdiss.value)
    torus_doppler = calc_doppler(Gamma,torus_radius.value,Rdiss.value)

    # AM3 photon grid
    egrid_jetframe = am3.get_egrid_photons() * u.eV

    # Set up array for adding up external fields
    external_photons = np.zeros(egrid_jetframe.size) * u.GeV / u.cm**3

    # Scattered thermal disk emission
    disky = ShakuraFlux(egrid_jetframe / blr_doppler,
                        Ld,
                        M,
                        thetaobs = 3.0) # erg/s, black hole frame

    # Broad line emission
    hybl = BroadLine(egrid_jetframe / blr_doppler,
                        10.2*u.eV, 10.2*u.eV/20,
                        Ld * blr_covering_factor) # H Ly alpha [erg/s]
    # hybl = BroadLineBB(egrid_jetframe / blr_doppler, # use this one if you want blackbody shape of BLR instead
    #                     10.2*u.eV,
    #                     Ld * blr_covering_factor) # H Ly alpha [erg/s]
    # Convert BLR components to jet frame
    blr_to_jet = convert_lum_to_density_in_jet(Rdiss.value,Gamma,blr_radius.value) # erg/s -> GeV/cm3 # there is a bug here
    blr_to_jet *= u.GeV / u.cm ** 3 / (u.erg / u.s) # give it units
    external_photons += (hybl) * blr_to_jet # GeV/cm3

    # Dust torus
    torusy = PlanckDistribution(egrid_jetframe / torus_doppler,
                                torus_temperature,
                                Ld * torus_covering_factor) # [erg/s]
    # Convert torus emission to jet frame
    tor_to_jet = convert_lum_to_density_in_jet(Rdiss.value,Gamma,torus_radius.value) # erg/s -> GeV/cm3
    tor_to_jet *= u.GeV / u.cm**3 / (u.erg/u.s)  # give it units
    # Add torus to BLR components
    external_photons += torusy * tor_to_jet

    # Convert summed up components from energy density to photon density in jet frame
    external_photonspectrum = (external_photons / egrid_jetframe).to(u.cm ** -3).value # cm-3

    # Convert photon density to density ijnjection rate
    external_photonspectrum /= am3.get_escape_timescale() # cm-3.s-1

    # Finally, inject external photon spectrum into the simulation
    am3.set_injection_rate_photons(external_photonspectrum)
    ################################################################################

    ######################## Time evolution ##############################
    time = 0.
    while time < 3.0 * am3.get_escape_timescale(): # Run up to 3x the light-crossing time
        am3.evolve_step() # Evolve solver
        time += am3.get_solver_time_step() # Count time
    ############## Plot cooled electron spectrum vs. injected spectrum #########################
    dNdE_inj = (am3.get_injection_rate_electrons()*am3.get_escape_timescale()*4*np.pi/3*r_b**3).value
    dNdE = (am3.get_electrons()*(4*np.pi/3*r_b**3)).value
    egrid_electrons = am3.get_egrid_lep()
    dNdE_inj = dNdE_inj/egrid_electrons # move from EdN/dE to dN/dE
    dNdE = dNdE/egrid_electrons # move from EdN/dE to dN/dE
    plot_lognu_logF([(egrid_electrons/const.m_e.to(u.eV, u.mass_energy())).value*u.dimensionless_unscaled,
        (egrid_electrons/const.m_e.to(u.eV, u.mass_energy())).value*u.dimensionless_unscaled],
                        [dNdE/u.eV,dNdE_inj/u.eV],
                        ['-','--'],[1,2],['r','g'],
                        ['cooled electron density','injected electron density'],
                        'lorentz factor','dN/dE',
                        ylims=[1.0e+36,1.0e+49],xlims=[1.0,1.0e+6],multiplot=True,title=f"{sourcename} \n z={z}",
                        errorlist=None,filename=f'plots/spectra/electron/{sourcename}_electron_spectrum.png',showfig=False,grids=True)
    # The cooled spectrum should go like $\\gamma^{-2}$ before the break and $\\gamma^{-1-s_2}$ after the break.
    # write number of electrons injected and number of cooled electrons to file
    file_elspec = open(f"plots/spectra/electron/{sourcename}_electron_spectrum.txt", "w")
    file_elspec.write(f'N_inj = {np.trapz(dNdE_inj, x=egrid_electrons)}\n N_cool = {np.trapz(dNdE, x=egrid_electrons)}')
    file_elspec.close()
    ######################## Set up arrays for plotting the SED ####################################

    # Jet frame -> obs. frame
    energy_conversion = doppler / (1 + z)

    # erg/cm3, source frame -> erg/cm2/s, obs. frame
    density_to_lum = 4/3 * np.pi * am3.get_escape_timescale() ** 2 * const.c.cgs.value ** 3
    lum_to_flux = 1./(4 * np.pi * d_l.value ** 2)
    spectrum_conversion = density_to_lum * lum_to_flux * doppler ** 4

    # Energy arrays in source frame
    egrid_pho = am3.get_egrid_photons()
    egrid_nu = am3.get_egrid_neutrinos()

    # Energy arrays in observer's frame
    egrid_pho_obs = egrid_pho * energy_conversion
    egrid_nu_obs = egrid_nu * energy_conversion

    # Get individual SED components
    all_nu = am3.get_neutrinos() * egrid_nu * u.eV.to(u.erg) * spectrum_conversion # all neutrino flavors
    nu_lum  = np.trapz(am3.get_neutrinos(),egrid_nu)*u.eV.to(u.erg)*density_to_lum # bolometric neutrino luminosity in erg/s
    file_nu.write(f"{sourcename} & ${nu_lum:.2e}$ & ${P_i_p.value:.2e}$ & ${P_i.value:.2e}$ & ${Ld.value:.2e}$ \\\\\n") # make latex table of neutrino luminosity, electron luminosity, proton luminosity and disk luminosity
    electron_syn = am3.get_photons_injected_electrons_syn() * egrid_pho * u.eV.to(u.erg) * spectrum_conversion
    electron_com = am3.get_photons_injected_electrons_compton() * egrid_pho * u.eV.to(u.erg) * spectrum_conversion
    annihil = am3.get_photons_annihilation_pairs_syn_compton() * egrid_pho * u.eV.to(u.erg) * spectrum_conversion
    bheitler = am3.get_photons_bethe_heitler_pairs_syn_compton() * egrid_pho * u.eV.to(u.erg) * spectrum_conversion
    pgamma = am3.get_photons_photo_pion_pairs_syn_compton() * egrid_pho * u.eV.to(u.erg) * spectrum_conversion
    pi0decay = am3.get_photons_pi0_decay() * egrid_pho * u.eV.to(u.erg) * spectrum_conversion
    proton_syn_ic = am3.get_photons_protons_syn_compton() * egrid_pho * u.eV.to(u.erg) * spectrum_conversion
    # Plot thermal emission
    ethermal = np.logspace(-3,8,100)
    eobs = ethermal / (1 + z)
    disk = ShakuraFlux(ethermal * u.eV,
                        Ld,
                        M
                        ) * lum_to_flux

    torus = PlanckDistribution(ethermal * u.eV,
                                torus_temperature,
                                Ld * torus_covering_factor
                                ) * lum_to_flux
    corona = Corona(ethermal* u.eV,corona_minimum_energy,corona_maximum_energy,corona_spectral_index,Ld*fx,d_l,disky,ethermal)
    if sourcename == "1149-084":
        corona = 0.1*corona # this particular source has a much weaker corona

    # sum over all radiation
    summ_e, summ_sed = summ_spectra(eobs, disk.value+torus.value+corona.value, egrid_pho_obs, electron_syn, nbin=500)
    summ_e, summ_sed = summ_spectra(summ_e,summ_sed,egrid_pho_obs, electron_com, nbin=500)
    summ_e, summ_sed = summ_spectra(summ_e,summ_sed,egrid_pho_obs, annihil, nbin=500)
    summ_e, summ_sed = summ_spectra(summ_e,summ_sed,egrid_pho_obs, bheitler, nbin=500)
    summ_e, summ_sed = summ_spectra(summ_e,summ_sed,egrid_pho_obs, pgamma, nbin=500)
    summ_e, summ_sed = summ_spectra(summ_e,summ_sed,egrid_pho_obs, pi0decay, nbin=500)
    summ_e, summ_sed = summ_spectra(summ_e,summ_sed,egrid_pho_obs, proton_syn_ic, nbin=500)

    gamma_lum = np.trapz(am3.get_photons(),egrid_pho)*u.eV.to(u.erg)*density_to_lum* doppler ** 4
    file_gamma.write(f"{sourcename}; {gamma_lum:.2e}\n") # print luminosity of jet EM radiation in observer frame
    #

    ############################# Data points from Ghisellini et al. (2011) ########################################
    # errorbars found by using WebPlotDigitizer
    # https://apps.automeris.io/wpd/
    data = np.loadtxt(f'errordata/{sourcename}.txt',delimiter = "; ",dtype=str)
    frequency_to_energy = 4.1357*10**(-15) # planck constant in eV*second
    errorbar_dic ={data[0,3]: [],data[1,3]: [],data[2,3]: [],data[3,3]: [],data[4,3]: []}

    data_arrows = np.loadtxt(f'errordata/{sourcename}_arrows.txt',delimiter = "; ",dtype=str)
    arrow_dic = {data_arrows[0,3]: [],data_arrows[1,3]: [],data_arrows[2,3]: []}
    fill_dictionaries(data,errorbar_dic)
    fill_dictionaries(data_arrows,arrow_dic)
    data_ovals = np.loadtxt(f'errordata/{sourcename}_ovals.txt',delimiter = "; ",dtype=str)
    if list(data_ovals) == []:# skips over NED data if text file is empty
        errorlist = [[errorbar_dic['Median'][:,0]*frequency_to_energy,errorbar_dic['Median'][:,1],
                     (errorbar_dic['Confidence right'][:,0]-errorbar_dic['Confidence left'][:,0])*0.5*frequency_to_energy,
                      (errorbar_dic['Standard error +1'][:,1]-errorbar_dic['Standard error -1'][:,1])*0.5,
                       '.','Errorbars (Swift)'],
                     [(arrow_dic['Confidence right'][:,0]+arrow_dic['Confidence left'][:,0])*0.5*frequency_to_energy,
                      arrow_dic['Median'][:,1],
                     (arrow_dic['Confidence right'][:,0]-arrow_dic['Confidence left'][:,0])*0.5*frequency_to_energy,
                      None,11,'Upper limits (Swift)']
                     ]
    else:
        oval_dic = {data_ovals[0,3]: []}
        fill_dictionaries(data_ovals,oval_dic)

        errorlist = [[errorbar_dic['Median'][:,0]*frequency_to_energy,errorbar_dic['Median'][:,1],
                     (errorbar_dic['Confidence right'][:,0]-errorbar_dic['Confidence left'][:,0])*0.5*frequency_to_energy,
                      (errorbar_dic['Standard error +1'][:,1]-errorbar_dic['Standard error -1'][:,1])*0.5,
                       '.','Simultaneous data (Ghisellini)'],
                     [(arrow_dic['Confidence right'][:,0]+arrow_dic['Confidence left'][:,0])*0.5*frequency_to_energy,
                      arrow_dic['Median'][:,1],
                     (arrow_dic['Confidence right'][:,0]-arrow_dic['Confidence left'][:,0])*0.5*frequency_to_energy,
                      None,11,'Upper limits (Ghisellini)'],
                     [oval_dic['Median'][:,0]*frequency_to_energy,oval_dic['Median'][:,1],
                      None,None,"o",'NED archive data (Ghisellini)']]

    ######################### 4FGL-DRE error bars ##################################
    FGL_earr = np.logspace(8,12,20) # energy array, eV # 100 MeV to 1 TeV
    ev_to_erg = 1.60217663e-12
    FGL_norm = float(input_4FGL[i+1,1]) # photon flux photons/cm^2/s/MeV
    FGL_norm_err = float(input_4FGL[i+1,2]) # photon flux error photons/cm²/s/MeV
    FGL_pivot = float(input_4FGL[i+1,5])*10**6 # pivot energy # given in MeV on HEASARC website
    FGL_index = -float(input_4FGL[i+1,3]) # spectral index
    FGL_index_err = float(input_4FGL[i+1,4]) # spectral index error
    FGL_y1 = FGL_earr**2*10**(-6)*ev_to_erg*FGL_norm*(FGL_earr/FGL_pivot)**FGL_index*(1+((FGL_norm_err/FGL_norm)**2+(FGL_index_err*np.log(FGL_earr/FGL_pivot))**2)**(0.5))
    FGL_y2 = FGL_earr**2*10**(-6)*ev_to_erg*FGL_norm*(FGL_earr/FGL_pivot)**FGL_index*(1-((FGL_norm_err/FGL_norm)**2+(FGL_index_err*np.log(FGL_earr/FGL_pivot))**2)**(0.5))
    FGL_color = 'c'
    FGL_label = '4FGL data'
    FGL_shaded_region = [FGL_earr,FGL_y1,FGL_y2,FGL_color,FGL_label]

    ######################### Neutrino sensitivity #################################
    nu_arr = np.logspace(12,15,10)*u.eV # energy array
    nu_sens = (3*nu_arr[0]**2*np.ones(len(nu_arr))*float(input_neutrino[i+1,1])/(u.TeV*u.cm**2*u.s)).to(u.erg/(u.cm**2*u.s)) # factor three due to flavour composition
    nu_sens_future = (3*np.ones(len(nu_arr))*float(input_neutrino[i+1,2])*u.TeV/(u.cm**2*u.s)).to(u.erg/(u.cm**2*u.s)) # factor three due to flavour composition
    nu_sens_arca = (3*np.ones(len(nu_arr))*float(input_neutrino[i+1,3])*u.GeV/(u.cm**2*u.s)).to(u.erg/(u.cm**2*u.s)) # factor three due to flavour composition # arca sensitivity in units of per GeV, not per TeV like in IceCube
    file_nu_barplot.write(f"{sourcename}; {np.amax(all_nu)/np.amax(nu_sens_future.value)}; {np.amax(all_nu)/np.amax(nu_sens_arca.value)}\n") # save the ratio between peak neutrino flux and flux sensitivity for use in barplot

    ######################### Plot the SED #################################################
    plot_lognu_logF([eobs*u.eV,egrid_pho_obs*u.eV,egrid_pho_obs*u.eV,egrid_pho_obs*u.eV,
                    egrid_pho_obs*u.eV,egrid_pho_obs*u.eV,egrid_pho_obs*u.eV,egrid_pho_obs*u.eV,
                    egrid_nu_obs*u.eV,
                    summ_e*u.eV,nu_arr,nu_arr,nu_arr],
                    [disk.value*u.erg/(u.cm**2*u.s) + torus.value*u.erg/(u.cm**2*u.s)+corona,
                        electron_syn*u.erg/(u.cm**2*u.s),electron_com*u.erg/(u.cm**2*u.s),
                        annihil*u.erg/(u.cm**2*u.s),
                        bheitler*u.erg/(u.cm**2*u.s),
                        pgamma*u.erg/(u.cm**2*u.s),
                        pi0decay*u.erg/(u.cm**2*u.s),
                        proton_syn_ic*u.erg/(u.cm**2*u.s),
                        all_nu*u.erg/(u.cm**2*u.s),
                        summ_sed*u.erg/(u.cm**2*u.s),nu_sens,nu_sens_future,nu_sens_arca],
                    ['--','-.','-',':',
                        ':','-.','--','-','-',
                        '-','--','-.',':'],
                    [2,2,2,2,
                        2,2,2,2,2,
                        4,2,2,2],
                    ['r','g','g','g',
                        'b','b','y','b','m',
                        'k','m','m','m'],
                    ['Thermal','Primary electrons (syn)','Primary electrons (IC)','gamma-gamma pairs',
                        'Bethe-Heitler pairs','Proton-gamma pions and leptons',
                        '$\pi^0$ decay','Proton synchrotron + IC',
                        'Neutrinos (all flavours)',
                        'Total photons','Neutrino 5$\sigma$ discovery potential','Future neutrino 5$\sigma$ discovery potential','Neutrino sensitivity of ARCA230'],
                    'Energy','Energy flux',
                    ylims = [1.0e-15,1.0e-10],xlims = [None,1.0e+21],
                    multiplot=True,title = f"{sourcename} \n z={z}",
                    errorlist=errorlist,filename = f'plots/leptohadronic/leptohadronic_{sourcename}.png',
                    showfig=False,
                    shaded_region=FGL_shaded_region,
                    tight_layout=True,
                    outside_legend=True,
                    figuresize = (18,10))
    print(f'Plot {i+1} of {number_of_sources} done.\n')
    am3.clear_am3() # clear all arrays and internal arrays
file_nu.close() # close file
file_nu_barplot.close() # close file
file_gamma.close() # close file
