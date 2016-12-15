# Folder structure 

A brief summary of the folder structure 

### Results 
Where all the results are stored. 
* Density: I am still in the process of adding in core contributions.. 
* Efield: Most of the results on E-field have been re-calculated here. Note whenever it says "anharmonic" for permittivities it means that all the MC samples are reweighted according to the anharmonic phonon distribution, and in addition, anharmonic frequencies are used instead of harmonic ones. The harmonic eigenvectors are still used as the difference they make is small. 

### InputFiles 
Note IceIh is used to refer to a specific IceIh_Cmc21 structure. The slight abuse of name is due to the fact that the effect of proton order is investigated after temperature dependence. 

### SamplingPrograms
Programs for generating the Monte Carlo samples and performing CASTEP calculations on them 

### Density
Where all the work on Density is stored. Results have been plotted excluding core contributions. 

### Efield 
Temperature dependence of three proton orders, using PBE functional. Here IceIh_Cmc21 refers to the specific structure, i.e. same as IceIh in InputFiles. 

### ProtonOrders
Ground state permittivities of more proton orders, using PBE functional. Here IceIh_Cmc21 refers to IceIh_Cmc21 in InputFiles.

### Functionals 
Functionals, using IceIh in InputFiles. 

### ZMapping 
Mapping of Zeff. The graphs are copied across to Results folder. 

### VSCF
VSCF solver. It was used to obtain the correction to vibrational frequencies due to E-field. A summary of the results of those run have been copied to Results folder. 

### Orbitals
Core orbitals. 

### CASTEP-8.0
The dark art of modifying CASTEP to output core electron densities. 

### Archive
Dumpster for previous versions of programs and testings. 

