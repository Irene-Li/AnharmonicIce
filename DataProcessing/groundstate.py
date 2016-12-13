import numpy as np 
from os import listdir 
from os.path import join, isfile, isdir 
from Readers import PermReader, DispReader
import h5py

class GroundStatePermittivity(object):

	def __init__(self, path):
		self.path = path
		self.directories = [join(path, x) for x in listdir(path) if isdir(join(path, x))]
		self.labels = [x for x in listdir(path) if isdir(join(path,x))]
		self.permReader = PermReader()

	def samplePermittivity(self):
		outfile = join(self.path, 'permittivity_summary.dat')
		f = open(outfile, 'w')
		startMode = 3

		electronPermAccu = 0 
		phononPermAccu = 0 

		electronPermSquaredAccu = 0 
		phononPermSquaredAccu = 0 
		totalPermSquaredAccu = 0 

		for (i, (directory, label)) in enumerate(zip(self.directories, self.labels)):

			castepFiles = [join(directory, x) for x in listdir(directory) if x.endswith('castep')]
			assert len(castepFiles) == 1
			permFile = castepFiles[0]
			dispFile = join(directory, 'disp_patterns.dat')
			self.permReader.read(permFile)
			dispReader = DispReader(self.permReader.nAtoms, self.permReader.nAtoms * 3 - 3)
			dispReader.read(dispFile)
			disp = dispReader.disp[startMode:]
			frequency = dispReader.frequency[startMode:]

			electronPerm = self.permReader.perms
			phononPerm = self.calculatePhononPerm(self.permReader.Zeffs, 
							disp, frequency, self.permReader.cellVolume)
			totalPerm = electronPerm + phononPerm
			self.writeSummary(f, label, electronPerm, phononPerm, totalPerm)

			if i != 0:
				electronPermAccu += electronPerm
				phononPermAccu += phononPerm
				electronPermSquaredAccu += electronPerm ** 2 
				phononPermSquaredAccu += phononPerm ** 2
				totalPermSquaredAccu += totalPerm ** 2 


		nOrders = len(self.labels) - 1 
		electronPermMean = electronPermAccu/nOrders
		phononPermMean = phononPermAccu/nOrders
		totalPermMean = electronPermMean + phononPermMean 
		electronPermStd = np.sqrt((nOrders * electronPermSquaredAccu - electronPermAccu ** 2)/
									(nOrders * (nOrders - 1)))
		phononPermStd = np.sqrt((nOrders * phononPermSquaredAccu - phononPermAccu ** 2)/
									(nOrders * (nOrders - 1)))
		totalPermStd = np.sqrt((nOrders * totalPermSquaredAccu - (totalPermMean * nOrders) ** 2)/
									(nOrders * (nOrders - 1)))

		self.writeSummary(f, 'Average', 
			electronPermMean, phononPermMean, totalPermMean)
		self.writeSummary(f, 'Standard Deviation', 
			electronPermStd, phononPermStd, totalPermStd)

		f.close()

	def calculatePhononPerm(self, Zeff, disp, frequency, cellVolume):
		frequencyFactor = 1/(frequency ** 2)
		osciStr = np.einsum('aij, maj, bkl, mbl -> mik', 
				            Zeff, disp, Zeff, disp)
		phononPerm = ((4 * np.pi / cellVolume)
					* np.einsum('ijk, i -> jk', osciStr, frequencyFactor)) 
		return phononPerm

	def writeSummary(self, fObject, label, electronContribution, phononContribution, total):

		# Write the header 
		fObject.write(' ===================== 龴ↀ◡ↀ龴 ===================== \n ')
		fObject.write('Label:	{} \n \n'.format(label))

		# Write the separate contributions
		fObject.write('Electron Contribution: \n')
		fObject.write(np.array2string(electronContribution, precision = 4))
		fObject.write('\n')
		fObject.write('Phonon Contribution: \n')
		fObject.write(np.array2string(phononContribution, precision = 4))
		fObject.write('\n')
		fObject.write('Permittivity: \n')
		fObject.write(np.array2string(total, precision = 4))
		fObject.write('\n')

		# Write the geometrical average of total permittivity
		eigenvalues = np.linalg.eigvals(total)
		fObject.write('Permittivities along principle axes: \n')
		fObject.write(np.array2string(eigenvalues))
		fObject.write('\n')
		fObject.write('Geometrical Average: \n [ {} ]'.format(np.prod(eigenvalues) ** (1/3)))
		fObject.write('\n')

		fObject.write('\n')

if __name__ == '__main__':

	'''
	path = 'ProtonOrders'
	tool = GroundStatePermittivity(path)
	tool.samplePermittivity()
	'''

	path = 'Functionals'
	tool = GroundStatePermittivity(path)
	tool.samplePermittivity()










