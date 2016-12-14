import numpy as np 
from matplotlib import pyplot as plt
from os import listdir 
from os.path import join, isfile
from readers import MappingReader, PermReader, EnergyReader, DispReader
import h5py

class Mapping(object):
	'''
	Class for mapping relative permittivites and Z-effs along some directions 

	Required structure of the directory: 
	-	files that end with the same string, which contain the quantities to be averaged over
		There must be a number in the middle to index the files. 
		e.g. IceIh.20.den_fmt for electron density
	-	mapping.out 
	Note that the required structure is automatically put into place by shell script mapping

	'''

	def __init__(self):
		self.endstring = 'castep' 
		self.permReader = PermReader()

	def addDirectory(self, directory):
		self.directory = directory
		self.files = [join(self.directory, x) for x in listdir(self.directory) if x.endswith(self.endstring)]
		self.nFiles = len(self.files)

		self.orderings = np.array([int(x.split('.')[-2]) for x in listdir(self.directory) if x.endswith(self.endstring)])
		self.ordeings, self.files = zip(*sorted(zip(self.orderings, self.files)))

		self.summary = join(self.directory, 'mapping.output')
		assert isfile(self.summary)

	def readSummary(self):
		mappingReader = MappingReader()
		mappingReader.read(self.summary)

		self.firstMode = mappingReader.firstMode
		self.lastMode = mappingReader.lastMode
		self.samplesPerMode = mappingReader.samplesPerMode
		self.nAtoms = mappingReader.nAtoms
		self.nStds = mappingReader.nStds
		self.double = mappingReader.double

	def mapAllModes2hdf5(self, update = False, deg = 3):
		self.readSummary()
		if self.double:
			self.getDoubleMode(plot = False, update = update)
		else:
			for mode in range(self.firstMode, self.lastMode+1):
				self.getSingleMode(mode, plot = False, update = update, deg = deg)

	def getSingleMode(self, mode, plot = False, update = False, deg = 3):
		assert mode <= self.lastMode and mode >= self.firstMode
		filename = join(self.directory,'SingleMode_{}.hdf5'.format(mode))
		if update or not isfile(filename):
			self.mapSingleMode(mode)
			self.fitSingleMode(deg)
			self.saveSingleMode2hdf5(mode)
		if plot:
			self.readSingleMode(mode, readRawData = True)
			self.plotSingleMode(mode)

	def getDoubleMode(self, plot = False, update = False):
		filename = join(self.directory, 'DoubleMode_{}and{}.hdf5'.format(self.firstMode, self.lastMode))
		if update or not isfile(filename):
			self.mapDoubleMode()
			self.saveDoubleMode2hdf5()
		if plot:
			self.readDoubleMode()
			self.plotDoubleMode()

	'''
	Hepler functions
	'''

	def mapDoubleMode(self): 
		self.perm = np.zeros((self.samplesPerMode, self.samplesPerMode, 3, 3))
		self.Zeff = np.zeros((self.samplesPerMode, self.samplesPerMode, self.nAtoms, 3, 3))

		i, j = map(np.ravel, np.meshgrid(np.arange(self.samplesPerMode, dtype = int), np.arange(self.samplesPerMode, dtype = int)))
		for n in range(self.nFiles):
			f = self.files[n]
			self.permReader.read(f)

			assert self.nAtoms == self.permReader.nAtoms
			self.perm[i[n], j[n]] = self.permReader.perms
			self.Zeff[i[n], j[n]] = self.permReader.Zeffs

	def plotDoubleMode(self):
		axis = np.arange(self.samplesPerMode)
		ticks = ['{0:.2f}'.format(x) for x in np.linspace(-self.nStds, self.nStds, self.samplesPerMode)]

		for k in range(self.nAtoms):
			plt.imshow(self.Zeff[:, :, k, 0, 0], cmap='viridis', interpolation='nearest')
			plt.xticks(axis, ticks)
			plt.yticks(axis, ticks)
			plt.xlabel('mode {}'.format(self.firstMode))
			plt.ylabel('mode {}'.format(self.lastMode))
			plt.colorbar()
			plt.title('Z* for Mode {} and {}, Atom No. {}'.format(self.firstMode, self.lastMode, k))
			plt.savefig(join(self.directory, 'figures', 'Zeff_mode{}and{}_atom{}.eps'.format(self.firstMode, self.lastMode, k)))
			plt.close()

		plt.imshow(self.perm[:, :, 0, 0], cmap='viridis', interpolation='nearest')
		plt.xticks(axis, ticks)
		plt.yticks(axis, ticks)
		plt.xlabel('mode {}'.format(self.firstMode))
		plt.ylabel('mode {}'.format(self.lastMode))
		plt.colorbar()
		plt.title('Permittivity for mode {} and {}'.format(self.firstMode, self.lastMode))
		plt.savefig(join(self.directory, 'figures', 'perm_mode{}and{}.eps'.format(self.firstMode, self.lastMode)))
		plt.close()

	def mapSingleMode(self, mode):
		self.perm = np.zeros((self.samplesPerMode, 3, 3))
		self.Zeff = np.zeros((self.samplesPerMode, self.nAtoms, 3, 3))

		start = (mode - self.firstMode) * self.samplesPerMode
		end = (mode - self.firstMode + 1) * self.samplesPerMode

		for i, n in enumerate(range(start, end)):
			f = self.files[n]
			self.permReader.read(f)

			assert self.nAtoms == self.permReader.nAtoms
			self.perm[i] = self.permReader.perms
			self.Zeff[i] = self.permReader.Zeffs 

	def fitSingleMode(self, deg):
		axis = np.linspace(-self.nStds, self.nStds, self.samplesPerMode)
		
		self.permFit = np.polyfit(axis, self.perm.reshape((self.samplesPerMode, 9)), deg)
		self.ZeffFit = np.polyfit(axis, self.Zeff.reshape((self.samplesPerMode, self.nAtoms * 9)), deg)
		self.permFit = self.permFit.reshape((deg+1, 3, 3))
		self.ZeffFit = self.ZeffFit.reshape((deg+1, self.nAtoms, 3, 3))		

	def plotSingleMode(self, mode):
		axis = np.linspace(-self.nStds, self.nStds, self.samplesPerMode)
		axis_forFitting = np.linspace(-self.nStds, self.nStds, 100)

		# Plot Z_eff 
		for i in range(self.nAtoms):
			plt.plot(axis, self.Zeff[:, i, 0, 0], 'r+', mew = 2, ms = 8, label = 'c-axis')
			plt.plot(axis, self.Zeff[:, i, 1, 1], 'g+', mew = 2, ms = 8, label = 'b-axis')
			plt.plot(axis, self.Zeff[:, i, 2, 2], 'b+', mew = 2, ms = 8, label = 'a-axis')
			plt.plot(axis_forFitting, np.poly1d(self.ZeffFit[:, i, 0, 0])(axis_forFitting), 'r--', label = 'c-axis fit')
			plt.plot(axis_forFitting, np.poly1d(self.ZeffFit[:, i, 1, 1])(axis_forFitting), 'g--', label = 'b-axis fit')
			plt.plot(axis_forFitting, np.poly1d(self.ZeffFit[:, i, 2, 2])(axis_forFitting), 'b--', label = 'a-axis fit')
			plt.title('Z* for Mode {}, Atom No. {}'.format(mode, i))
			plt.xlabel('Number of Standard Deviation')
			plt.ylabel('Z*')
			plt.legend()
			plt.savefig(join(self.directory, 'figures', 'Zeff_mode{}_atom{}.eps'.format(mode, i)))
			plt.close()

		# Plot permittivity
		plt.plot(axis, self.perm[:, 0, 0], 'r+', mew = 2, ms = 8, label = 'c-axis')
		plt.plot(axis, self.perm[:, 1, 1], 'g+', mew = 2, ms = 8, label = 'b-axis')
		plt.plot(axis, self.perm[:, 2, 2], 'b+', mew = 2, ms = 8, label = 'a-axis')
		plt.plot(axis_forFitting, np.poly1d(self.permFit[:, 0, 0])(axis_forFitting), 'r--', label = 'c-axis fit')
		plt.plot(axis_forFitting, np.poly1d(self.permFit[:, 1, 1])(axis_forFitting), 'g--', label = 'b-axis fit')
		plt.plot(axis_forFitting, np.poly1d(self.permFit[:, 2, 2])(axis_forFitting), 'b--', label = 'a-axis fit')
		plt.title('Permittivity for Mode {}'.format(mode))
		plt.xlabel('Number of Standard Deviation')
		plt.ylabel('Permittivity')
		plt.legend()
		plt.savefig(join(self.directory, 'figures', 'Perm_mode{}.eps'.format(mode)))
		plt.close() 

	def saveSingleMode2hdf5(self, mode):
		filename = join(self.directory,'SingleMode_{}.hdf5'.format(mode))
		hdf = h5py.File(filename, 'w')
		perm = hdf.create_dataset('permittivity', data = self.perm)
		Zeff = hdf.create_dataset('Zeff', data = self.Zeff)
		hdf.attrs['mode'] = mode
		hdf.attrs['nAtoms'] = self.nAtoms
		hdf.attrs['nStds'] = self.nStds
		hdf.attrs['samplesPerMode'] = self.samplesPerMode
		perm.attrs['fit'] = self.permFit
		Zeff.attrs['fit'] = self.ZeffFit
		hdf.close()

	def readSingleMode(self, mode, readRawData = False):
		if isinstance(mode, int):
			mode = join(self.directory,'SingleMode_{}.hdf5'.format(mode))
		else:
			assert isinstance(mode, str)
			assert 'SingleMode' in mode
			assert mode.endswith('hdf5')
		hdf = h5py.File(mode, 'r')
		self.nAtoms = hdf.attrs['nAtoms']
		self.nStds = hdf.attrs['nStds']
		self.permFit = hdf['permittivity'].attrs['fit'][()]
		self.ZeffFit = hdf['Zeff'].attrs['fit'][()]
		modeNumber = hdf.attrs['mode']

		if readRawData:
			self.samplesPerMode = hdf.attrs['samplesPerMode']
			self.perm = hdf['permittivity'][()]
			self.Zeff = hdf['Zeff'][()]

		hdf.close()

		return modeNumber

	def saveDoubleMode2hdf5(self):
		filename = join(self.directory,'DoubleMode_{}and{}.hdf5'.format(self.firstMode, self.lastMode))
		hdf = h5py.File(filename, 'w')
		hdf.create_dataset('permittivity', data = self.perm)
		hdf.create_dataset('Zeff', data = self.Zeff)
		hdf.attrs['modes'] = [self.firstMode, self.lastMode]
		hdf.attrs['nAtoms'] = self.nAtoms
		hdf.attrs['nStds'] = self.nStds 
		hdf.attrs['samplesPerMode'] = self.samplesPerMode
		hdf.close()

	def readDoubleMode(self):
		filename = join(self.directory, 'DoubleMode_{}and{}.hdf5'.format(self.firstMode, self.lastMode))
		assert isfile(filename)
		hdf = h5py.File(filename, 'r')
		self.nAtoms = hdf.attrs['nAtoms']
		self.nStds = hdf.attrs['nStds'] 
		self.samplesPerMode = hdf.attrs['samplesPerMode']
		self.perm = hdf['permittivity'][()]
		self.Zeff = hdf['Zeff'][()]
		hdf.close()

class MappingTool(object):
	'''
	A class that uses the mapping class to map all related directories 
	'''

	def __init__(self, dirs):
		self.dirs = dirs
		self.mapper = Mapping()	

	def mapAllDirectories(self, deg = 3):
		for directory in dirs:
			self.mapper.addDirectory(directory)
			self.mapper.mapAllModes2hdf5(update = False, deg = deg)

	def peep(self):
		self.mapper.addDirectory(self.dirs[0])
		self.mapper.readSummary()
		self.nAtoms = self.mapper.nAtoms
		self.nModes = self.nAtoms * 3 - 3 # excluding the translational ones

	def collect(self, terminal):
		filename = join(terminal, 'mapping.hdf5')
		hdf = h5py.File(filename, 'w')
		hdf.attrs['nAtoms'] = self.nAtoms
		hdf.attrs['nModes'] = self.nModes

		for directory in self.dirs:
			files = [join(directory, x) for x in listdir(directory) if x.startswith('SingleMode') and x.endswith('.hdf5')]
			for f in files:
				mode = str(self.mapper.readSingleMode(f))
				print('looking at mode:', mode)
				print('Maximum 1st order constant:', np.max(self.mapper.ZeffFit[-2]))
				assert self.nAtoms == self.mapper.nAtoms
				grp = hdf.create_group(mode)
				grp.create_dataset('ZeffFit', data = self.mapper.ZeffFit)
				grp.create_dataset('permFit', data = self.mapper.permFit)

		hdf.close() 

class Energy(object):
	'''
	A class that calculates the correction to BO surface due to 
	'''

	def __init__(self, wd):
		self.workingDirectory = wd
		self.energyFile = join(wd, 'energy.dat')
		self.dispFile = join(wd, 'disp_patterns.dat')
		self.ZeffFile = join(wd, 'mapping.hdf5')

	def calculatePolarisation(self):
		hdf = h5py.File(self.ZeffFile, 'r')

		self.nModes = hdf.attrs['nModes']
		self.nAtoms = hdf.attrs['nAtoms']
		self.modes = hdf.keys() 
		assert len(self.modes) == self.nModes # make sure that all the modes are mapped

		self.readEnergy()
		self.readDisp()

		self.effectivePolarisation = np.zeros((self.nModes, self.samplesPerMode, 3)) 
									# for 3 components of the electric field
		for mode in self.modes:
			index = int(mode) - 4 # counting in keys() starts from 4
			phononPrefractor = np.sqrt(2 * self.frequency[index]) # 1/std
			_, a, b, c = hdf[mode]['ZeffFit'][()] # 2nd, 1st and 0th 
			a *= (phononPrefractor ** 2) # shape = (nAtoms, 3, 3)
			b *= phononPrefractor # shape = (nAtoms, 3, 3)
			coord = self.phononCoord[index] # shape = (samplesPerMode)
			polarisation = np.einsum('ijk, m -> ijkm', a/3, coord ** 3) + np.einsum('ijk, m -> ijkm', 
							b/2, coord ** 2) + np.einsum('ijk, m -> ijkm', c, coord) 
							# shape = (nAtoms, 3, 3, samplesPerMode)
			self.effectivePolarisation[index] = np.einsum('nj, nijk -> ki', 
												self.disp[index], polarisation) 
												# shape = (samplesPerMode, 3)

		hdf.close()

	def calculateEnergy(self, Efield, filename):
		correction = np.einsum('mij, j-> mi', self.effectivePolarisation, Efield)
		self.BOSurface += correction # shape = (nModes, samplesPerMode)
		self.energyReader.write(self.BOSurface, filename)

	def readEnergy(self):
		self.energyReader = EnergyReader(self.nModes) # the first three are translational modes
		self.energyReader.read(self.energyFile)
		self.samplesPerMode = self.energyReader.samplesPerMode
		self.frequency = self.energyReader.frequency
		self.BOSurface = self.energyReader.BOSurface # shape = (nModes, nSamplesPerMode)
		self.phononCoord = self.energyReader.phononCoord # shape = (nModes, nSamplesPerMode)

	def readDisp(self):
		dispReader = DispReader(self.nAtoms, self.nModes)
		dispReader.read(self.dispFile)
		self.disp = dispReader.disp # shape = (nModes, nAtoms, 3)




if __name__ == '__main__':
	
	# dirs = [join('ZMapping', 'output', x) for x in ['single_1', 'single_2', 'single_3', 'single_4']]
	# terminal = join('ZMapping', 'output')
	# tool = MappingTool(dirs)
	# tool.peep()
	# tool.mapAllDirectories()
	# tool.collect(terminal)

	wd = 'VSCF/input_files'
	outfile = join(wd, 'modified_energy.dat')
	energy = Energy(wd)
	energy.calculatePolarisation()
	energy.calculateEnergy([0, 1, 0], outfile)
	






	


