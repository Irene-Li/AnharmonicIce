import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import Utils

'''
List of readers in this class 
	FMTReader (for .den_fmt)
	PermReader (for .castep)
	XSFReader (for .xsf)
	EfieldReader (for .efield)
	MappingReader (for mapping.output)
	EnergyReader (for energy.dat)
	DispReader (for disp_patterns.dat)
	AnhEigenvaluesReader (for anharmonic_eigenvalues.dat)
	CellReader (for .cell)
	OrbsReader (for .orbs) 
'''

class FMTReader(object): 
	'''
	Extract densities and the grid it's on from .den_fmt files 

	Useful class variables:
		realLatticeVectors: lattice vector in real space in Angstrom
		shape: shape of the grid on which density is stored 
		size: size of the density grid = np.prod(shape)
		densities: densities on a grid 
	'''

	def __init__(self):
		pass 

	def peep(self, filename):
		self.infoTable = pd.read_table(filename, header = None, nrows = 9)
		self.denTable = pd.read_table(filename, header = None, skiprows = 12, delim_whitespace=True)
		self.getInfo()

	def read(self, filename):
		self.infoTable = pd.read_table(filename, header = None, nrows = 9)
		self.denTable = pd.read_table(filename, header = None, skiprows = 12, delim_whitespace=True)

		self.getInfo()
		self.getDensity()

	def getInfo(self):
		col = 0 # Only 1 column in infoTable
		shapeIndex = 6
		realLatticeVectorStartIndex = 2

		# Initialise the realLatticeVectors array
		self.realLatticeVectors = np.zeros((3, 3))

		# Load the data into arrays
		self.shape = np.fromstring(self.infoTable[col][shapeIndex], dtype = int, sep = ' ')[0:3]
		self.size = np.prod(self.shape)

		for i in range(self.realLatticeVectors.shape[0]):
			self.realLatticeVectors[i] = np.fromstring(self.infoTable[col][realLatticeVectorStartIndex + i], dtype = float, sep = ' ')[0:3]

	def getDensity(self):
		'''
		reads pandas dataframe to arrays
		'''
		self.densities = (np.array(self.denTable.loc[:, 3])/self.size).reshape(self.shape, order = 'F')

	def getIndices(self):
		indices = np.array(self.denTable.loc[:, :2]) - 0.5
		return indices

	def CoM(self, size, densities, coordinates):

		flatDensities = densities.reshape((size), order = 'F')
		flatCoordinates = coordinates.reshape((size, 3), order = 'F')

		return np.sum(flatCoordinates.T * flatDensities, axis = -1)/np.sum(flatDensities)

class PermReader(object):
	'''
	Extracts permittivity etc from .castep files 

	Useful class variables: 
		permittivity
		polarisability
		BornEffectiveCharge: born effective charge? 
	'''
	def __init__(self):
		pass 

	def peep(self, filename):
		self.f = open(filename, 'r')
		self.getCellVolume()
		self.getAtomNumber()
		self.f.close()

	def read(self, filename): 
		self.f = open(filename, 'r') 
		# DO NOT CHANGE THE ORDER OF THE FUNCTIONS
		self.getCellVolume()
		self.getAtomNumber()
		self.getPerm()
		self.getZeff()
		self.f.close()

	def getCellVolume(self):
		line = ' '
		while 'Current cell volume' not in line:
			line = self.f.readline()
		lengthConversion = 1.8897259886 # from Angstrom to Bohr
		self.cellVolume = float(line.split()[-2]) * lengthConversion ** 3

	def getAtomNumber(self):
		line = ' '
		while 'Total number of ions in cell' not in line:
			line = self.f.readline()
		self.nAtoms = int(line.split()[-1])


	def getPerm(self):
		self.perms = np.zeros((3, 3))
		line = ' '
		while 'Optical Permittivity' not in line:
			line = self.f.readline()
		line = self.f.readline() # read an empty line 
		for i in range(3):
			line = self.f.readline()
			self.perms[i] = np.fromstring(line, dtype = 'float', sep = ' ')[0:3]

	def getZeff(self):
		self.Zeffs = np.zeros((self.nAtoms, 3, 3))
		line = ' ' 
		while 'Born Effective Charges' not in line:
			line = self.f.readline()
		line = self.f.readline()
		for i in range(self.nAtoms):
			line = self.f.readline()
			self.Zeffs[i, 0] = np.array([float(x) for x in line.split()[2:]])
			for j in range(1, 3):
				line = self.f.readline()
				self.Zeffs[i, j] = np.fromstring(line, dtype = 'float', sep = ' ')


class XSFReader(object):
	'''
	reads .xsf files to numpy arrays 

	Useful class variables: 
		realLatticeVectors: lattice vectors
		shape: shape of the grid
		densities: densities of the grid points, 3D numpy array
		size: size of the densities
	'''
	def __init__(self):
		pass 

	def peep(self, filename):
		self.infoTable = pd.read_table(filename, header = None, skiprows = 50, nrows = 5, delim_whitespace=True)
		self.getShape()


	# Main function
	def read(self, filename):
		self.infoTable = pd.read_table(filename, header = None, skiprows = 50, nrows = 5, delim_whitespace=True)
		self.getShape()

		self.denTable = pd.read_table(filename, header = None, skiprows = 55, nrows = self.size, delim_whitespace=True)
		self.getDensities()
		self.modifyDensities()

	def getShape(self):
		# Write down the positions all the rows we want 
		shapeIndex = 0
		realLatticeVectorStartIndex = 2

		# Load the data into arrays
		self.shape = np.array(self.infoTable.loc[shapeIndex, :], dtype = np.int)
		self.realLatticeVectors= np.array(self.infoTable.loc[2:5, :]) 
		self.size = np.prod(self.shape)

	def getDensities(self):
		# Load the data into arrays
		self.densities = np.array(self.denTable).reshape(self.shape, order = 'F')
		assert self.size == self.densities.size

	def modifyDensities(self):
		'''
		reorder the axis: 0, 1, 2 -> 2, 1, 0
		'''
		self.densities = np.swapaxes(self.densities, 0, 2)[0:-1, 0:-1, 0:-1]
		self.shape = np.flipud(self.shape) - 1
		self.realLatticeVectors = np.flipud(self.realLatticeVectors)
		self.size = np.prod(self.shape)

class EfieldReader(object):

	def __init__(self):
		pass 
		
	def peep(self, filename):
		f = open(filename, 'r')
		self.getInfo(f)
		f.close()

	def read(self, filename):
		self.peep(filename) # in case the information changes
		self.getOsciStr(filename)
		self.getPerms(filename)

	def getInfo(self, fObject):
		line = ' '
		while 'Number of ions' not in line:
			line = fObject.readline()
		self.nIons = int(line.split()[-1])

		line = fObject.readline()
		assert 'Number of branches' in line
		self.nBranches = int(line.split()[-1])

		line = fObject.readline()
		assert 'Number of frequencies' in line
		self.nFreqs = int(line.split()[-1])

		line = fObject.readline()
		assert 'Oscillator Q' in line
		self.Q = float(line.split()[-1])

	def getOsciStr(self, filename):
		oscillatorStartRow = 12 + self.nIons
		osciStrTable = pd.read_table(filename, header=None, skiprows=oscillatorStartRow, 
									nrows=self.nBranches, delim_whitespace=True)
		self.osciFreq = np.array(osciStrTable.loc[:, 1], dtype=np.float)
		self.osciStr = np.array(osciStrTable.loc[:, 2:], dtype=np.float)

	def getPerms(self, filename):
		permStartRow = 14 + self.nIons + self.nBranches
		permTable = pd.read_table(filename, header=None, skiprows=permStartRow, 
								  nrows=self.nFreqs, delim_whitespace=True)
		self.freq = np.array(permTable.loc[:, 0], dtype=np.float)
		self.perm = np.array(permTable.loc[:, 1:], dtype=np.float)

	# Quick Plot function 
	def plotPerm(self): # plot E_perpendicular
		plt.plot(self.freq[self.perm[:, 0] < 4], self.perm[:, 0][self.perm[:, 0] < 4])
		plt.title('permittivity along the c-axis')
		plt.xlim([2000, 4000])
		plt.xlabel('wavelength')
		plt.ylabel('relative permittivity')
		plt.show()

class MappingReader(object):
	'''
	reads mapping.output
	'''

	def __init__(self):
		self.startKeyWord = 'Number of basis atoms'

	def read(self, filename):
		f = open(filename, 'r')
		line = ' '
		while self.startKeyWord not in line:
			line = f.readline()
		self.nAtoms = int(line.split()[-1])

		line = f.readline()
		assert 'First mode' in line
		self.firstMode = int(line.split()[-1])

		line = f.readline()
		assert 'Last mode' in line
		self.lastMode = int(line.split()[-1])

		line = f.readline()
		assert 'Samples per mode' in line
		self.samplesPerMode = int(line.split()[-1])

		line = f.readline()
		assert 'Double' in line
		self.double = (line.split()[-1].lower() == 'true')

		line = f.readline()
		line = f.readline()
		assert 'Number of stds' in line
		self.nStds = int(line.split()[-1])

class EnergyReader(object):
	'''
	reads energy.dat
	'''

	def __init__(self, modes):
		self.modes = modes

	def read(self, filename): # note this does not read frequencies
		self.filename = filename
		self.countSamplesPerMode()
		self.readFrequencies()
		self.readBOSurface()

	def countSamplesPerMode(self):
		f = open(self.filename, 'r')
		line = f.readline()
		count = 0
		while line.strip():
			if not line.startswith('#'):
				count += 1
			line = f.readline()
		self.samplesPerMode = count

	def readFrequencies(self):
		self.frequency = np.zeros((self.modes))
		reader = pd.read_table(self.filename, header = None, iterator = True, chunksize = self.samplesPerMode + 1)
		for (i, chunk) in enumerate(reader):
			self.frequency[i] = float(chunk.iloc[0, 0].split()[-1])

	def readBOSurface(self):
		self.phononCoord = np.zeros((self.modes, self.samplesPerMode))
		self.BOSurface = np.zeros((self.modes, self.samplesPerMode))
		reader = pd.read_table(self.filename, header = None, iterator = True, chunksize = self.samplesPerMode, comment = '#', delimiter = ',')
		for (i, chunk) in enumerate(reader):
			self.phononCoord[i] = np.array(chunk.iloc[:, 0])
			self.BOSurface[i] = np.array(chunk.iloc[:, 1])

	def write(self, BOSurface, filename): # modify the BOSurface without changing the frequencies and sampling points
		f = open(filename, 'w') 
		for i in range(self.modes):
			f.write('# {} \n'.format(self.frequency[i]))
			for j in range(self.samplesPerMode):
				f.write('{}, {} \n'.format(self.phononCoord[i, j], self.BOSurface[i, j]))
			f.write('\n')
			f.write('\n')
		f.close()

class DispReader(object):
	'''
	reads disp_pattern.dat
	'''
	def __init__(self, nAtoms, modes):
		self.modes = modes
		self.nAtoms = nAtoms

	def read(self, filename):
		self.frequency = np.zeros((self.modes))
		self.disp = np.zeros((self.modes, self.nAtoms, 3))
		f = open(filename, 'r')
		line = ''
		for i in range(self.modes):
			line = f.readline()
			self.frequency[i] = float(line.split()[-1])
			line = f.readline() # k point
			line = f.readline() # 'Displacement pattern for each atom:'
			for j in range(self.nAtoms):
				line = f.readline()
				self.disp[i, j, :] = np.fromstring(line, dtype = float, sep = ' ')[0:-1]
			line = f.readline() # blank line

class AnhEigenvaluesReader(object):
	'''
	reads anharmonic_eigenvalues.dat
	'''
	def __init__(self, nModes):
		self.nModes = nModes

	def read(self, filename):
		self.frequency = np.zeros((self.nModes))
		f = open(filename, 'r')
		line = ' '
		for i in range(self.nModes):
			while 'Eigenvalues for mode' not in line:
				line = f.readline()
			line = f.readline()
			self.frequency[i] = float(line) * 2 # ground state energy = omega/2


class CellReader(object):
	'''
	Reads -out.cell file to numpy array
	'''

	def peep(self, filename):
		ionTable = pd.read_table(filename, header = None, skiprows = 11, nrows = 36, 
									delim_whitespace = True) # in fractional coordinates
		self.species = np.array(ionTable[0])

	def read(self, filename):
		# Read from .cell files
		ionTable = pd.read_table(filename, header = None, skiprows = 11, nrows = 36, 
									delim_whitespace = True) # in fractional coordinates
		latticeTable = pd.read_table(filename, header = None, skiprows = 5, nrows = 3, 
										delim_whitespace = True)
		self.species = np.array(ionTable[0])

		# read positions to Cartesian
		self.latticeVectors = np.array(latticeTable, dtype = float)
		self.fracPositions = np.array(ionTable.loc[:, 1:], dtype = float)

class OrbsReader(object):

	# Can be used to read out an arbitrary number of orbits 
	def read(self, filename):
		table = pd.read_table(filename, header = None, delim_whitespace = True)
		n = np.array(table.loc[:, 0], dtype = int)
		l = np.array(table.loc[:, 1], dtype = int)
		radius = np.array(table.loc[:, 2], dtype = float)
		rPhi = np.array(table.loc[:, 3], dtype = float) # r * phi(r)

		self.nGridPoints = (np.sum(n == 1)) # count the number of grid points per orbital
		self.nOrbitals = int(n.size/(self.nGridPoints)) # count the number of orbitals 
		assert self.nOrbitals == Utils.calculateTotalNumberOfOrbitals(n[-1], l[-1])

		self.radius = radius[0:self.nGridPoints]
		self.rPhi = rPhi.reshape(self.nOrbitals, self.nGridPoints)







