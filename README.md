Leptohadronic.py is the main function that does the calculations and the plots. It requires AM3 to be installed.
Change the path in line 19 to where AM3 is stored on your computer.
I am not particularly good at programming, so you can probably make changes to optimize the code considerably.

The errrordata folder includes datapoints taken from the plots in the ghisellini_plots folder using a web plot digitizer.

Output plots are stored in the plots folder.

The file input_parameters.txt contains all the input parameters from Ghisellini et al. (2011)

The file proton_max_energy.txt contains the maximum lorentz factor of the proton spectrum for all the sources.

The file particlepowers.txt contains the baryon loading factor in its third column. Its second column can be changed to adjust the electron luminosity relative to that in input_parameters.txt (currently set to 1.0 for all sources).

The file neutrino_sensitivity.txt contains the flux sensitivity of both IceCube and KM3NeT/ARCA.

The file 4fgl.txt contains data from the HEASARC archive used to calculate the 4FGL bands.

The file 4fgk_alias.txt contains the 4FGL aliases of all the sources.

Further details are found in the Method chapter of my thesis.
