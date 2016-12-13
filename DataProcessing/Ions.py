import numpy as np
import scipy
from os.path import join 
from os import listdir 
from matplotlib import pyplot as plt
import Utils
from Readers import CellReader, OrbsReader
'''
TODO: 
	1. find out how scipy finds the best fit function, and decay length
'''
class Element(object):

	def __init__(self, name):
		self.name = name 
		self.coreWavefunctionStored = False 

	def add1s(self, function, decayLength):
		'''
		function: the core wavefunction
		decayLength: the length at which the contribution from the core 
						wave function becomes negligible
		'''
		self.coreWavefunctionStored = True
		self.coreDensity = function
		self.coreDecayLength = decayLength

class Ions(object):
	'''
	A class for calculation of bands
	Important class members:
		elements: a dictionary with names as keys, pointing to Element objects storing mass and core orbitals
		fracPositions: fractional positions of ions 
		realPositions: real positions of ions 
		latticeVectors: lattice vectors 
		elementNames: element names of the ions 
	'''		

	def __init__(self, **kwargs):
		self.cellReader = CellReader()
		if 'names' in kwargs:
			names = set(kwargs['names'])
		elif 'cellFile' in kwargs:
			filename = kwargs['cellFile']
			self.cellReader.peep(filename)
			names = set(self.cellReader.species)
		else:
			raise Exception("Unexpected Argument")
		self.elements = {name : Element(name) for name in names}

	def getCoreOrbits(self, elementName, orbsFile):
		coreWavefunction, decayLength = self.get1S(orbsFile)
		self.elements[elementName].add1s(coreWavefunction, decayLength)

	def getFracPositions(self, filename):
		# extract information from the file
		self.cellReader.read(filename)
		self.fracPositions = self.cellReader.fracPositions
		self.latticeVectors = self.cellReader.latticeVectors
		self.elementNames = self.cellReader.species

		# process the information
		self.getMasks()
		self.modifyIonCoordinates()


	# Helper functions 
	# ===============================================================================

	def getMasks(self): 
		'''
		only applied to the static cell so we could use the same transformation on all the cells
		'''
		# Masks for ions off the edge 
		self.smallerThanZero = (self.fracPositions < 0)
		self.largerThanOne = (self.fracPositions > 1) 

		# Masks for ions on the edge 
		self.closeToZero = np.isclose(self.fracPositions, 0, atol = 1e-4)
		self.closeToOne = np.isclose(self.fracPositions, 1, atol = 1e-4)
		self.rowsCloseToZero = np.any(self.closeToZero, axis = 1)
		self.rowsCloseToOne = np.any(self.closeToOne, axis = 1)

		# Only want the marks for the selected rows 
		self.closeToZero = self.closeToZero[self.rowsCloseToZero]
		self.closeToOne = self.closeToOne[self.rowsCloseToOne]

		# Modify the masks for ions off the edge so they don't include those that are too close
		self.smallerThanZero[self.rowsCloseToZero] = False
		self.largerThanOne[self.rowsCloseToOne] = False

	def modifyIonCoordinates(self):
		''' 
		CoM in cartersian coordinates 
		'''	
		self.fracPositions[self.smallerThanZero] = self.fracPositions[self.smallerThanZero] + 1 
		self.fracPositions[self.largerThanOne] = self.fracPositions[self.largerThanOne] - 1 
		
		ionsToAppend1 = self.fracPositions[self.rowsCloseToZero]
		ionsToAppend1[self.closeToZero] = ionsToAppend1[self.closeToZero] + 1 
		nameToAppend1 = self.elementNames[self.rowsCloseToZero]
		ionsToAppend2 = self.fracPositions[self.rowsCloseToOne]
		ionsToAppend2[self.closeToOne] = ionsToAppend2[self.closeToOne] - 1 
		nameToAppend2 = self.elementNames[self.rowsCloseToOne]

		self.fracPositions = np.concatenate((self.fracPositions, ionsToAppend1, ionsToAppend2), axis=0)
		self.elementNames = np.concatenate((self.elementNames, nameToAppend1, nameToAppend2), axis=0)								

	def plotIonAlongCAxis(self):
		for name in self.elements.keys():
			positions = Utils.frac2real(self.fracPositions[self.elementNames == name], self.latticeVectors)
			plt.scatter(positions[:, 0], positions[:, 1], label=name)
		Utils.plotCellFrameInABPlane(self.latticeVectors)
		plt.legend()
		plt.show()


	def get1S(self, orbsFile, plot=False): # onlt 1s implemented
		orbsReader = OrbsReader()
		orbsReader.read(orbsFile)

		# Extract the 1s orbit and the radial function
		rPhi = orbsReader.rPhi[0] # only take the first orbit, i.e. 1s 
		radius = orbsReader.radius

		# Check whether the normalisation works
		norms = Utils.integrateRadialWavefunction(radius, rPhi)
		tol = 1e-6 
		assert np.all(np.abs(norms - 1) < tol)

		# Find the best-fit function for 1s of the form phi(r) = Ae^(-kr)
		def bestfit(x, A, m):
			return A * np.exp(-m * x)

		(A, m), _ = scipy.optimize.curve_fit(bestfit, radius[1:], rPhi[1:]/radius[1:])
		decayLength = (7/m)
		print(decayLength)

		plt.plot((radius[1:]/decayLength), rPhi[1:]/radius[1:])
		plt.plot((radius[1:]/decayLength), bestfit(radius[1:], A, m))
		plt.xlim([0, 1.5])
		plt.show() 

		return (lambda x: bestfit(x, A, m)), decayLength


if __name__ == '__main__':

	cellFile = 'Density/output/output_static/IceIh-out.cell'
	orbsFile = 'orbitals/O_OTF.orbs'
	elements = ['H', 'O'] 
	ions = Ions(names=elements)
	ions.getFracPositions(cellFile)
	ions.plotIonAlongCAxis()

	ions.getCoreOrbits('O', orbsFile)










	