import numpy as np 
from matplotlib import pyplot as plt
from os import listdir 
from os.path import join, isfile
from scipy import stats
import Utils 
from Sampling import DenSampling
from Ions import Ions

class DensityAnalysisTool(object):
	'''
	Look at the average output density across several directories and compare them
	'''
	def __init__(self, dirs, tile, reciShape):
		# Add dirs and temps as class variables
		assert dirs # make sure that the list is not empty 
		self.dirs = dirs 
		self.labels = ['_'.join((x.split('/')[-1]).split('_')[1:]) for x in dirs]
		self.ndirs = len(self.dirs)

		# Store tile and reciShape
		self.tile = np.array(tile)
		self.reciShape = np.array(reciShape)

		# Initialise denSampling
		self.denSampling = DenSampling()

		# Extract basic parameters from the first directories in dirs
		self.denSampling.addDirectory(self.dirs[0])
		self.denSampling.initialise(reciShape = self.reciShape, tile = self.tile)
		self.shape = self.denSampling.shape
		self.realLatticeVectors = self.denSampling.realLatticeVectors
		self.reciprocalLatticeVectors = self.denSampling.reciprocalLatticeVectors
		self.realCoordinates = self.denSampling.realCoordinates
		self.reciCoordinates = self.denSampling.reciCoordinates

	def getDensities(self, nPlots = 0, update = False, anharmonicity = False):
		self.realDensities = np.zeros((self.ndirs, *self.shape))
		self.reciDensities = np.zeros((self.ndirs, *self.reciShape))

		for n in range(self.ndirs):

			d = self.dirs[n]
			self.denSampling.addDirectory(d)
			self.denSampling.average(update = update, save = True, 
									anharmonicity = anharmonicity)

			self.realDensities[n] = self.denSampling.realMean
			self.reciDensities[n] = self.denSampling.reciMean

			print(self.labels[n])
			print(self.denSampling.sum())
				
			if nPlots > 0:
				self.denSampling.plotAverageRealDensities(nPlots)
				self.denSampling.plotAverageReciDensities(nPlots)
				

	def compareDensities(self, index1, index2, real = True, log = False, frac = False, path = '.', nodoge = True):
		title = '({} - {})'.format(self.labels[index1], self.labels[index2])

		if real:
			diff = (self.realDensities[index1] - self.realDensities[index2])
			coor = self.realCoordinates
			cIndex = 0
			figurename = join(path, 'figures', 'realDiff_{}vs{}_{}'.format(self.labels[index1], self.labels[index2], cIndex))
		else:
			diff = (self.reciDensities[index1] - self.reciDensities[index2])
			coor = self.reciCoordinates
			cIndex = int(self.reciShape[-1]/2)
			figurename = join(path, 'figures', 'reciDiff_{}vs{}_{}'.format(self.labels[index1], self.labels[index2], cIndex))

		if frac:
			title = '{}/{}'.format(title, self.labels[index2])
			if real:
				diff /= self.realDensities[index2]
			else:
				diff /= self.reciDensities[index2]

		if log:
			diff = np.log(diff)
			title = 'log({})'.format(title)

		Utils.plotAlongCAxis(diff, coor, figurename, index = cIndex, sum = False, title = title, contour = False)

		if not nodoge:
			print(' ─────────▄──────────────▄──── \n', \
			 	  ' ─ wow ──▌▒█───────────▄▀▒▌─── \n', \
			  	  ' ────────▌▒▒▀▄───────▄▀▒▒▒▐─── \n', \
			  	  ' ───────▐▄▀▒▒▀▀▀▀▄▄▄▀▒▒▒▒▒▐─── \n', \
			  	  ' ─────▄▄▀▒▒▒▒▒▒▒▒▒▒▒█▒▒▄█▒▐─── \n', \
			  	  ' ───▄▀▒▒▒▒▒▒ such difference ─ \n', \
			  	  ' ──▐▒▒▒▄▄▄▒▒▒▒▒▒▒▒▒▒▒▒▒▀▄▒▒▌── \n', \
			  	  ' ──▌▒▒▐▄█▀▒▒▒▒▄▀█▄▒▒▒▒▒▒▒█▒▐── \n', \
			  	  ' ─▐▒▒▒▒▒▒▒▒▒▒▒▌██▀▒▒▒▒▒▒▒▒▀▄▌─ \n', \
			  	  ' ─▌▒▀▄██▄▒▒▒▒▒▒▒▒▒▒▒░░░░▒▒▒▒▌─ \n', \
			  	  ' ─▌▀▐▄█▄█▌▄▒▀▒▒▒▒▒▒░░░░░░▒▒▒▐─ \n', \
			  	  ' ▐▒▀▐▀▐▀▒▒▄▄▒▄▒▒▒ electrons ▒▌ \n', \
			  	  ' ▐▒▒▒▀▀▄▄▒▒▒▄▒▒▒▒▒▒░░░░░░▒▒▒▐─ \n', \
			  	  ' ─▌▒▒▒▒▒▒▀▀▀▒▒▒▒▒▒▒▒░░░░▒▒▒▒▌─ \n', \
			  	  ' ─▐▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▐── \n', \
			  	  ' ──▀ amaze ▒▒▒▒▒▒▒▒▒▒▒▄▒▒▒▒▌── \n', \
			  	  ' ────▀▄▒▒▒▒▒▒▒▒▒▒▄▄▄▀▒▒▒▒▄▀─── \n', \
			  	  ' ───▐▀▒▀▄▄▄▄▄▄▀▀▀▒▒▒▒▒▄▄▀───── \n', \
			  	  ' ──▐▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▀▀──────── \n'
				)



if __name__ == '__main__':

	# Initialisation
	dirs = ['output_static', 'output_0K', 'output_260K']
	dirs = [join('Density/output', x) for x in dirs]
	path = 'output'
	tile = (4, 4, 1)
	reciShape = (64, 64, 60)
	tool = DensityAnalysisTool(dirs, tile, reciShape)

	# Get densities from the file
	tool.getDensities(update = False, nPlots = 10, anharmonicity = False)

	# Compare densities
	# for i in range(1, len(dirs)):
	# 	tool.compareDensities(i, 0, real = True, frac = False, log = False, path = path)
	# 	tool.compareDensities(i, 0, real = False, frac = False, log = False, path = path)
# 
	# tool.compareDensities(2, 1, real = True, frac = False, log = False, path = path)
	# tool.compareDensities(2, 1, real = False, frac = False, log = False, path = path)




