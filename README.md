# Vibrational correction to Ice

General purpose Python codes for averaging over Monte Carlo samples in vibrational analysis once they are obtained from CASTEP calculations. See the content of the files for more detailed documentations.

Here is a brief summary of what each file does. 

### efield.py 
Contains class PermAnalysisTool for calculating and plotting permittivities in a few related directories (e.g. different temperatures, different proton orders, different functionals). See the main method for examples of how it is used. 

### density.py
Contains class DensityAnalysisTool for calculating and plotting densities in a few related directories (e.g. different temperatures). See the main method for examples. 

### mapping.py
For mapping Z* and electronic permittivity over vibrational normal modes

### groundstate.py 
For comparing the ground state permittivities of different proton orders 

### Sampling.py
Classes for Monte Carlo sampling over castep outputs to be used in efield.py and density.py
* MCSampling: abstract class, providing abstract functions to be implemented by subclasses
* DenSampling: one implementation of MCSampling for averaging over densities, in both real space and reciprocal space, with additional plotting functions
* Bands: one implementation of MCSampling for plotting densities of energy bands
* PermSampling: one implementation of MCSampling for averaging over permittivities, with additional plotting functions 

### Readers.py
Readers for extracting various inputs from files. 
* FMTReader (for .den_fmt)
* PermReader (for .castep)
* XSFReader (for .xsf)
* EfieldReader (for .efield)
* MappingReader (for mapping.output)
* EnergyReader (for energy.dat)
* DispReader (for disp_patterns.dat)
* AnhEigenvaluesReader (for anharmonic_eigenvalues.dat)
* CellReader (for .cell)
* OrbsReader (for .orbs) 

### Ions.py 
Ions related functions to add core electron contributions to densities. 

### Utils.py
Useful general purpose functions for other classes



