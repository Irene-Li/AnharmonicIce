import numpy as np 
from readers import DispReader, AnhEigenvaluesReader

dispFile = 'Efield/input_files/disp_patterns.dat'
anhFile = 'Efield/input_files/anharmonic_eigenvalues.dat'
nAtoms = 24
nModes = nAtoms * 3

reader = DispReader(nAtoms, nModes)
reader.read(dispFile)
harFreq = reader.frequency[3:]

reader = AnhEigenvaluesReader(nModes-3)
reader.read(anhFile)
anhFreq = reader.frequency

for i in range(len(harFreq)):
	print('{:.6f}	{:.6f}'.format(harFreq[i], anhFreq[i]))
