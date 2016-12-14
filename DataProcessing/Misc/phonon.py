import numpy as np 
from readers import DispReader, PermReader, EfieldReader

# Read in eigenvalues 
nAtoms = 24 
nModes = nAtoms * 3
filename = 'Efield/input_files/disp_patterns.dat'
reader = DispReader(nAtoms, nModes)
reader.read(filename) 
disp = reader.disp

# Masses 
mHydrogen = 1.837362123816600E+03
mOxygen = 2.916512050696600E+04
mass = np.zeros((nAtoms)) # shape = (nAtoms)
mass[0:16] = mHydrogen
mass[16:] = mOxygen

# Unit cell volumns, in A
A = [4.396186, -0.000004, 0.000000]
B = [0.000007, 7.652359, -0.003143]
C = [0.000000, -0.002212, 7.188613] 
lengthConversion = 1.8897259886 # from Angstrom to Bohr
volume = np.dot(A, np.cross(B, C)) * lengthConversion ** 3


# Read in Bohr effective charges 
reader = PermReader()
filename = 'Efield/output/static/IceIh.1.castep'
reader.read(filename)
Zeff = reader.Zeffs # shape = (nAtoms, 3, 3)
perm = reader.perms # shape = (3, 3)
indices = [(0, 1, 2, 1, 2, 0), (0, 1, 2, 2, 0, 1)]
perm = perm[indices]

# Reader in oscillator strength and frequencies 
reader = EfieldReader()
filename = 'Efield/output/static/IceIh.efield'
reader.read(filename)
freqConversion = 4.55633E-6 # from cm-1 to Hartree (a.u.)
massConversion = 1.82289E3 # from amu to electron mass (a.u.)
chargeConversion =  0.20819434 # from Debye/Angstrom (WTF) to e (a.u.)
frequency = reader.osciFreq * freqConversion # shape = (nModes)
strength = reader.osciStr * chargeConversion ** 2/massConversion # shape = (nModes, 3, 3)

staticPerm = reader.perm[0]

# Orthogonality check 
print('Orthonality check')
print(np.einsum('i, ij, ij', mass, disp[4], disp[5]), '\n')

# Normalisation condition check 
print('Normalisation check')
print(np.einsum('i, ij, ij', mass, disp[4], disp[4]), '\n')

# check 
factor = (1/frequency)**2
osciStr = np.einsum('aij, maj, bkl, mbl -> mik', Zeff, disp, Zeff, disp)
phononContribution = 4 * np.pi * np.einsum('ijk, i -> jk', osciStr[3:], factor[3:]) / volume
phononContribution = phononContribution[indices]
print(factor)
print('electron contribution: ')
print(perm)
print('phonon contribution: ')
print(phononContribution)
print('calculated from disp_patterns and Zeff: ')
print(phononContribution + perm)
print('output from castep phonon+efield: ')
print(staticPerm)
print('\n')

	
	







