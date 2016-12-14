import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from os import listdir 
from os.path import join, isfile
from scipy import stats
import h5py
import abc

from Readers import *
from Ions import Ions
import Utils
'''
TODO: 
	write a function in DenSampling that loops through all the elements and add its core wavefunction
'''

class MCSampling(object): # abstract class
	'''
	General purpose MC-sampling parent class for averaging over all the files in a directory with weights

	Required structure of the directory: 
	  -	files that end with the same string, which contain the quantities to be averaged over
		There must be a number in the middle to index the files. 
		e.g. IceIh.20.den_fmt for electron density
	  - weights.dat that contain the weights of all the data points. 
		There is one-to-one correspondence between lines in weights.dat. For example, the ith row of weights.dat 
		corresponds to IceIh.i.den_fmt
	Note that the required structure is automatically put into place by shell scripts "caesar" and "mapping"

	'''

	def __init__(self):
		self.endstring = '' # need to be overwritten in subclasses 

	def addDirectory(self, directory):
		'''
		Find all the files in directory that end with self.endstring and order them according to filenames
		'''
		self.directory = directory
		self.files = [join(self.directory, x) for x in listdir(self.directory) 
					  							if x.endswith(self.endstring)]
		self.nFiles = len(self.files)

		self.orderings = np.array([int(x.split('.')[-2]) for x in listdir(self.directory) 
								   						if x.endswith(self.endstring)])
		self.orderings, self.files = zip(*sorted(zip(self.orderings, self.files)))


	@abc.abstractmethod
	def initialise(self):
		'''
		Open one file and extract shape parameters from the file
		Mainly useful for initialising data storage for density/permittivities
		This is not part of __init__ because the initialisation can only be done once a directory has been added
		'''
		pass

	@abc.abstractmethod
	def average(self, update = False, save = True, anharmonicity = False):
		'''
		Average over all the files in a directory, if ahharmonic, use weights from weights.dat to average
		'''
		pass 

	def getWeights(self, anharmonicity):
		'''
		Get weights from weights.dat if anharmonicty is true, otherwise generate an array of ones
		'''
		if anharmonicity:
			weightFile = join(self.directory, 'weights.dat')
			self.weights = np.array(pd.read_table(weightFile, header = None))
			assert self.weights.size >= self.nFiles
			assert self.weights.size >= self.orderings[-1]
			self.weights = self.weights[0:self.nFiles]
		else:
			self.weights = np.ones((self.nFiles))

class PermSampling(MCSampling):
	'''
	'''

	def __init__(self):
		self.endstring = '.castep'
		self.permReader = PermReader()

	def initialise(self, dispFile):
		self.permReader.peep(self.files[0])
		self.nAtoms = self.permReader.nAtoms
		self.cellVolume = self.permReader.cellVolume
		nModes = self.nAtoms * 3
		startMode = 3

		dispReader = DispReader(self.nAtoms, nModes)
		dispReader.read(dispFile)
		self.disp = dispReader.disp[startMode:]
		self.frequencyFractor = 1/(dispReader.frequency[startMode:] ** 2) 

	def average(self, update, save, anharmonicity, anhFreqFile = '', numberOfFiles = 0):
		self.getWeights(anharmonicity)

		if numberOfFiles == 0:
			numberOfFiles = self.nFiles

		if anharmonicity:
			label = 'AveragePerms_anh_{}'.format(numberOfFiles)
			self.anhFreq(anhFreqFile)
		else:
			label = 'AveragePerms_har_{}'.format(numberOfFiles)

		if (isfile(join(self.directory, '{}.hdf5'.format(label))) and (not update)):
			nAtoms = self.nAtoms
			self.readHDF5(label)
			# the information stored is not the same as what we need, recalculate for the file
			if nAtoms != self.nAtoms:
				self.calculateAveragePerms(numberOfFiles)
				if save:
					self.saveAveragePerms2hdf5(label)
		else:
			self.calculateAveragePerms(numberOfFiles)
			if save:
				self.saveAveragePerms2hdf5(label)

	def readPerm(self, fileNumber):
		assert fileNumber < self.nFiles
		f = self.files[fileNumber]
		self.permReader.read(f)
		assert self.nAtoms == self.permReader.nAtoms 
		electronPerm = self.permReader.perms 
		osciStr = np.einsum('aij, maj, bkl, mbl -> mik', 
			                 self.permReader.Zeffs, self.disp, self.permReader.Zeffs, self.disp)
		phononPerm = (4 * np.pi * np.einsum('ijk, i -> jk', osciStr, self.frequencyFractor) 
							/ self.cellVolume)
		return electronPerm, phononPerm

	def anhFreq(self, anhFreqFile):
		eigenvalueReader = AnhEigenvaluesReader(self.nAtoms * 3 - 3) # minus the three translational modes
		eigenvalueReader.read(anhFreqFile)
		self.frequencyFractor = 1/(eigenvalueReader.frequency ** 2)

	'''
	Helper functions
	'''

	def calculateAveragePerms(self, numberOfFiles):
		assert numberOfFiles <= self.nFiles
		self.electronPerm = np.zeros((3, 3))
		self.phononPerm = np.zeros((3, 3))
		
		total_weight = 0


		for n in range(numberOfFiles):
			f = self.files[n]
			self.permReader.read(f)
			assert self.nAtoms == self.permReader.nAtoms
			self.electronPerm += self.permReader.perms * self.weights[n]
			osciStr = np.einsum('aij, maj, bkl, mbl -> mik', 
				                 self.permReader.Zeffs, self.disp, self.permReader.Zeffs, self.disp)
			self.phononPerm += (self.weights[n] * 4 * np.pi 
								* np.einsum('ijk, i -> jk', osciStr, self.frequencyFractor) 
								/ self.cellVolume)
			total_weight += self.weights[n]

		self.electronPerm /= total_weight
		self.phononPerm /= total_weight
		self.staticPerm = self.electronPerm + self.phononPerm

	def saveAveragePerms2hdf5(self, label):
		filename = '{}.hdf5'.format(join(self.directory, label))
		hdf = h5py.File(filename, 'w')
		hdf.create_dataset('electron permittivity', data = self.electronPerm)
		hdf.create_dataset('phonon permittivity', data = self.phononPerm)
		hdf.create_dataset('total permittivity', data = self.staticPerm)
		hdf.attrs['nAtoms'] = self.nAtoms
		hdf.close() 

	def readHDF5(self, label):
		filename = '{}.hdf5'.format(join(self.directory, label))
		hdf = h5py.File(filename, 'r')
		self.nAtoms = hdf.attrs['nAtoms']
		self.electronPerm = hdf['electron permittivity'][()]
		self.phononPerm = hdf['phonon permittivity'][()]
		self.staticPerm = hdf['total permittivity'][()]
		hdf.close()

class DenSampling(MCSampling):

	def __init__(self):
		self.endstring = 'den_fmt'
		self.fmtReader = FMTReader()

	def initialise(self, reciShape = (5, 5, 5), tile = (1, 1, 1)):
		'''
		Get basic parameters: shape, lattice vectors, reciprocal lattice vectors, coordinates and reciprocal coordinates
		options:
			- reciShape: number of cells in reciprocal space
			- tile: how many unit cells to fourier transform 
		'''
		# Initialise from den_fmt file 
		self.fmtReader.peep(self.files[0])
		self.shape = self.fmtReader.shape
		self.reciShape = reciShape
		self.tile = tile
		self.realLatticeVectors = self.fmtReader.realLatticeVectors
		self.reciprocalLatticeVectors = Utils.getReciprocalLatticeVector(self.realLatticeVectors)

		# Generate the coordinates
		indices = self.fmtReader.getIndices()
		self.realCoordinates = Utils.fracs2Coordinates(indices/self.shape, 
														self.realLatticeVectors, self.shape)
		self.reciCoordinates = Utils.shape2Coordinates(reciShape, self.reciprocalLatticeVectors, 
														tile, shift = True) 
														# tiled and shifted coordinates
																					

	def initialiseCoreOrbs(self, elementName, orbsFile):	
		# Initialise the ion lattice
		cellFile = Utils.denfmt2cell(self.files[0])
		self.ions = Ions(cellFile = cellFile)
		self.ions.getCoreOrbits(elementName, orbsFile)

	def average(self, update = False, save = True, anharmonicity = False):
		self.getWeights(anharmonicity)
		self.label = Utils.getAnharmonicityLabel(anharmonicity)
		filename = 'AverageDensities_{}'.format(self.label)

		if (isfile(join(self.directory, '{}.hdf5'.format(filename))) and (not update)):
			# Store tile and reciShape for comparison
			tile = self.tile
			reciShape = self.reciShape
			self.readHDF5(filename)

			# the information stored is not the same as what we need, recalculate for the file
			if np.any(self.tile != tile) or np.any(self.reciShape != reciShape):
				self.calculateAverageDensities()
				if save:
					self.saveAverageDensities2hdf5(filename)
		else:
			self.calculateAverageDensities()
			if save:
				self.saveAverageDensities2hdf5(filename)

	def sum(self):
		# cellVolume = np.linalg.det(self.realLatticeVectors) * ()
		return np.sum(self.realMean)

	'''
	Plotting functions
	'''
	def plotAverageRealDensities(self, nPlots = 1):
		nlines = 10
		for index in range(0, self.shape[-1], int(self.shape[-1]/nPlots)):

			meanName = join(self.directory, 'figures/{}_RealMean_{}'.format(self.label, index))
			meanTitle = 'Mean of Real Lattice, Slice No. {}'.format(index)
			Utils.plotAlongCAxis(self.realMean, self.realCoordinates, meanName, 
								title=meanTitle, index=index, nlines=nlines)

			if self.calculateVar: 
				stdName = join(self.directory, 'figures/{}_RealStd_{}'.format(self.label, index))
				stdTitle = 'Std of Real Lattice, Slice No. {}'.format(index)
				Utils.plotAlongCAxis(np.sqrt(self.realVar), self.realCoordinates, stdName, 
									title=stdTitle, index=index, nlines=nlines)

	def plotAverageReciDensities(self, nPlots = 2):
		for index in range(0, self.reciShape[-1], int(self.reciShape[-1]/nPlots)):
			meanName = join(self.directory, 'figures/{}_ReciMean_{}'.format(self.label, index))
			meanTitle = 'Mean of Reciprocal Lattice, Slice No. {}'.format(index)
			angleName = join(self.directory, 'figures/{}_angle_{}'.format(self.label, index))
			angleTitle = 'Complex Argument in Reciprocal Space, Slice No. {}'.format(index)

			Utils.plotAlongCAxis(self.reciMean, self.reciCoordinates, meanName, title=meanTitle, 
									index=index, contour=False)
			Utils.plotAlongCAxis(self.reciMeanAngle, self.reciCoordinates, angleName, 
									title=angleTitle, index=index, contour=False)

			if self.calculateVar:
				stdname = join(self.directory, 'figures/{}_ReciStd_{}'.format(self.label, index))
				stdTitle = 'Std of Reciprocal Lattice, Slice No. {}'.format(index)
				Utils.plotAlongCAxis(np.sqrt(self.reciVar), self.reciCoordinates, stdName, 
										title=stdTitle, index=index, contour=False)

	def runningAverage(self, slices = 2):
		'''
		Perform a running average with different stopping point
		NEED TO BE UPDATED TO INCLUDE WEIGHTS
		'''
		self.realDensities = np.zeros((slices, *self.shape))
		self.reciDensities = np.zeros((slices, *self.reciShape))
		self.angle = np.zeros((slices, *self.reciShape))
		tempReciDensities = np.zeros((slices, *self.reciShape), dtype = 'complex128')

		start = 0
		self.count = np.full((slices), self.nFiles, dtype = 'int')
		for n in range(self.nFiles):
			f = self.files[n]
			densities = self.readDenFiles(f)

			# Record how many files are averaged over in "count"
			temp = start
			start = np.int(n/self.nFiles * slices)
			if start > temp:
				self.count[temp] = n

			self.realDensities[start:] += densities
			tempReciDensities[start:] += Utils.fft(densities, outshape = self.reciShape, 
													tile = self.tile)
		
		for i in range(slices):
			self.realDensities[i]/=self.count[i]
			self.reciDensities[i] = np.absolute(tempReciDensities[i]/self.count[i])
			self.angle[i] = np.angle(tempReciDensities[i]/self.count[i])

		self.realMean = self.realDensities[-1]
		self.reciMean = self.reciDensities[-1]
		self.reciMeanAngle = self.angle[-1]

	def chisquare(self, plot = False):
		'''
		Calculate variance of a variable number of samples and see if it's converged
		Require: runningAverage with at least 1 slice
		'''
		nSamples = self.realDensities.shape[0] - 1
		chisquare = np.zeros((nSamples))
		pvalues = np.zeros((nSamples))
		exp = self.realDensities[-1]

		for i in range(nSamples):
			obs = self.realDensities[i]
			chisquare[i], pvalues[i] = stats.chisquare(f_obs = obs, f_exp = exp, axis = None)
		

		if plot:
			plt.plot(self.count[:-1], chisquare, linestyle='--', marker='o', color='k')
			plt.title('Chisquare')
			plt.xlabel('Number of Samples')
			plt.ylabel('Chisquare')
			plt.yscale('log')
			plt.savefig(join(self.directory,'figures/chisquare.eps'))
			plt.close()

			plt.plot(self.count[:-1], pvalues, linestyle='--', marker='o', color='k')
			plt.title('P-value')
			plt.xlabel('Number of Samples')
			plt.ylabel('P-value')
			plt.yscale('log')
			plt.savefig(join(self.directory,'figures/pvalue.eps'))
			plt.close()

		return chisquare, pvalues

	'''
	Helper Functions
	'''
	def calculateAverageDensities(self, calculateVar = False):
		self.realMean = np.zeros(self.shape)
		self.reciMean = np.zeros(self.reciShape).astype('complex128')
		self.reciMeanAngle = np.zeros(self.reciShape)
		self.calculateVar = calculateVar
		total_weight = 0

		if calculateVar:
			self.realVar = np.zeros(self.shape)
			self.reciVar = np.zeros(self.reciShape)

		for n in range(self.nFiles):
			f = self.files[n]
			densities = self.readDenFiles(f)

			self.realMean += densities * self.weights[n]
			self.reciMean += Utils.fft(densities, outshape = self.reciShape, 
										tile = self.tile) * self.weights[n]
			total_weight += self.weights[n]

			if calculateVar:
				self.realVar += densities ** 2 * self.weights[n]
				self.reciVar += np.absolute(self.reciMean) ** 2 * self.weights[n]

		self.realMean /= total_weight
		self.reciMeanAngle = np.angle(self.reciMean/total_weight)
		self.reciMean = (np.absolute(self.reciMean/total_weight)).astype('float')
	
		if calculateVar:
			self.realVar /= total_weight
			self.reciVar /= total_weight

	def readDenFiles(self, denfile):
		'''

		'''
		self.fmtReader.read(denfile)
		assert np.all(self.shape == self.fmtReader.shape)
		assert np.all(self.realLatticeVectors - self.fmtReader.realLatticeVectors < 1e-6) 
		densities = self.fmtReader.densities

	def addAllCoreContributions(self, densities):
		cellFile = Utils.denfmt2cell(self.denfile)
		self.ions.getIonCoordinates(cellFile)

		for (i, name) in enumerate(self.ions.elementNames):
			
			if self.ions.elements[name].coreWavefunctionStored:

				fracPosition = self.ions.fracPositions[i] 
				length = self.ions.elements[name].coreDecayLength

				gridSpan = self.findGridSpan(length)
				gridPoints = self.findNearbyGridPoints(fracPosition, gridSpan)
				distances = np.apply_along_axis((lambda x: self.findDistance(x, fracPosition)), 
					1, gridPoints)
				coreDensities = self.ions.elements[name].coreDensities(distances)
				densities = self.addCoreDensities(coreDensities, gridPoints, densities)

	def addCoreDensities(self, coreDensities, gridPoints, densities):
		coreDensities /= np.sum(coreDensities) * 2 # normalisation 
		for (i, gridPoint) in enumerate(gridPoints):
			densities[gridPoint] = densities[gridPoint] + coreDensities[i]
		return densities

	def findGridSpan(self, length):
		'''
		find out the most number of grids that a certain length spans 
		'''
		minSeparation = min(np.linalg.norm(self.realLatticeVectors, axis = -1)/self.shape)
		return np.ceil(length/minSeparation)

	def findDistance(self, gridpoint, fracPosition):
		distance = (self.realCoordinates[gridpoint] - 
					Utils.frac2real(fracPosition, self.realLatticeVectors))
		return distance



	def findNearbyGridPoints(self, fracPosition, gridSpan):
		mid_i, mid_j, mid_k = np.floor(fracPositions * self.shape) # element-wise multiplication
		gridPoints = np.array([[i, j, k] for i in range(mid_i-gridSpan, mid_i+gridSpan+1)
								for j in range(mid_j-gridSpan, mid_j+gridSpan+1)
								for k in range(mid_k-gridSpan, mid_k+gridSpan+1)])
		print(gridPoints.shape)
		return gridPoints
							

	def saveAverageDensities2hdf5(self, label): # store the data in hdf5 format
		filename = '{}.hdf5'.format(join(self.directory, label))
		hdf = h5py.File(filename, 'w')
		real = hdf.create_group('real')
		reci = hdf.create_group('reci')

		real.create_dataset('densities', data = self.realMean)
		real.create_dataset('coor', data = self.realCoordinates)
		real.create_dataset('latticevectors', data = self.realLatticeVectors)
		real.attrs['shape'] = self.shape 

		reci.create_dataset('densities', data = self.reciMean)
		reci.create_dataset('angles', data = self.reciMeanAngle)
		reci.create_dataset('coor', data = self.reciCoordinates)
		reci.create_dataset('latticevectors', data = self.reciprocalLatticeVectors)
		reci.attrs['shape'] = self.reciShape
		reci.attrs['tile'] = self.tile

		if self.calculateVar:
			real.create_dataset('var', data = self.realVar)
			reci.create_dataset('var', data = self.reciVar)

		hdf.close()

	def readHDF5(self, label):
		filename = '{}.hdf5'.format(join(self.directory, label))
		hdf = h5py.File(filename, 'r')
		self.calculateVar = False

		real = hdf['real']
		reci = hdf['reci']

		self.realMean = real['densities'][()]
		self.realCoordinates = real['coor'][()]
		self.realLatticeVectors = real['latticevectors'][()]
		self.shape = real.attrs['shape'][()]

		self.reciMean = reci['densities'][()]
		self.reciMeanAngle = reci['angles'][()]
		self.reciCoordinates = reci['coor'][()]
		self.reciprocalLatticeVectors = reci['latticevectors'][()]
		self.reciShape = reci.attrs['shape'][()]
		self.tile = reci.attrs['tile'][()]

		if 'var' in real.keys():
			self.calculateVar = True
			self.realVar = real['var'][()]
			self.reciVar = reci['var'][()]

		hdf.close() 

class Bands(MCSampling):
	'''
	A class for calculation of bands
	'''

	def __init__(self, directory):
		super().__init__()
		self.xsfReader = XSFReader() 


	def addDirectory(self, directory):
		super().super().addDirectory(directory)
		
		self.xsfFiles = [join(self.directory, x) for x in listdir(self.directory) 
						if x.endswith('.xsf')]
		self.xsfBands = np.array([int(x.split('.')[-2]) for x in listdir(self.directory) 
									if x.endswith('.xsf')])

		# sort the two lists
		self.xsfBands, self.xsfFiles = zip(*sorted(zip(self.xsfBands, self.xsfFiles)))


	def getElectronDensityForBands(self, bandIndex): # To get all the bands, set bands = 0
		filename = self.xsfFiles[bandIndex] 
		self.xsfReader.read(filename)

		assert np.all(self.shape == self.xsfReader.shape)
		assert np.all(self.realLatticeVectors - self.xsfReader.realLatticeVectors < 1e-6)

		return self.xsfReader.densities

	def plotBands(self, nbands, totalElectronNumber, plotOrbits = False):
		densities = 0 
		for bandIndex in range(1, nbands+1):
			orbit = self.getElectronDensityForBands(bandIndex)
			densities += orbit/np.sum(orbit) * 2 # two electrons per orbit

			if plotOrbits:
				figureName = join(self.directory, 
									'figures/band_{}'.format(self.xsfBands[bandIndex]))
				title = 'Electron Density in the {}th band'.format(self.xsfBands[bandIndex])
				Utils.plotAlongCAxis(orbit/np.sum(orbit) * 2, self.realCoordinates, figureName, 
									title = title, contour = True, nlines = 8)

		figureName = join(self.directory, 'figures/With_{}_bands'.format(nbands))
		title = 'Electron Density in {} bands'.format(nbands)	
		Utils.plotAlongCAxis(densities, self.realCoordinates, figureName, 
							title = title, sum = False, contour = True, nlines = 8)

		leftoverDensities = self.realMean/np.sum(self.realMean) * totalElectronNumber - densities
		figureName = join(self.directory, 'figures/Without_{}_bands'.format(nbands))
		title = 'Electron Density without {} bands'.format(nbands)
		Utils.plotAlongCAxis(leftoverDensities, self.realCoordinates, figureName, 
							title = title, sum = False, contour = True, nlines = 8)


if __name__ == '__main__':
	directory = 'Density/output/output_0K'
	element = 'O'

	sampling = DenSampling(directory)















	















