import numpy as np 
import itertools
from matplotlib import pyplot as plt
from os import listdir 
from os.path import join, isfile
import Utils
from Sampling import PermSampling

class PermAnalysisTool(object):

	def __init__(self, dirs, dispFiles):
		# Add dirs and temps as class variables
		assert dirs # make sure that the list is not empty 
		self.dirs = dirs 
		self.disps = dispFiles
		self.labels = [', '.join(x.split('/')[1:]) for x in dirs]
		self.ndirs = len(self.dirs)
		assert self.ndirs == len(self.disps)

		# Initialise denSampling
		self.permSampling = PermSampling()

	def plotSamplePerms(self, numberOfFiles, terminal, anharmonicity=False, anhFreqFiles=[], index=(0, 0)):

		if anharmonicity:
			figname = 'permittivity_anh_{}_({}_{}).pdf'.format(numberOfFiles, index[0], index[1])
		else:
			anhFreqFiles = [''] * self.ndirs
			figname = 'permittivity_har_{}_({}_{}).pdf'.format(numberOfFiles, index[0], index[1])

		self.findPerms(numberOfFiles, anharmonicity, anhFreqFiles, index)

		fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
		markercycle = itertools.cycle(('.', 'x', '+', '*', 's'))

		for n in range(self.ndirs):
			label = self.dirs[n].split('/')[-3]
			marker = next(markercycle)
			ax1.plot(self.electronPerms[n], marker=marker, mew=2, ls='--', lw=1.5, label=label, alpha=0.7)
			ax2.plot(self.phononPerms[n], marker=marker, mew=2, ls='--', lw=1.5, label=label, alpha=0.7)
		ax1.legend(loc='lower right', prop={'size':10})
		ax2.legend(loc='upper right', prop={'size':10})
		ax1.set_ylabel('Electron Permittivity')
		ax2.set_ylabel('Phonon Permittivity')
		ax2.set_xlabel('Sample Number')
		ax1.set_title('Electron and phonon permittivity matrix element ({}, {})'.format(index[0], index[1]))
		plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
		plt.savefig(join(terminal, figname))
		plt.close()

	def findPerms(self, numberOfFiles, anharmonicity, anhFreqFiles, index):

		self.electronPerms = np.zeros((self.ndirs, numberOfFiles))
		self.phononPerms = np.zeros((self.ndirs, numberOfFiles))


		for n in range(self.ndirs):
			d = self.dirs[n] 
			disp = self.disps[n] 
			self.permSampling.addDirectory(d)
			self.permSampling.initialise(disp)

			if anharmonicity:
				self.permSampling.anhFreq(anhFreqFiles[n])

			for i in range(numberOfFiles):
				ele, pho = self.permSampling.readPerm(i)
				self.electronPerms[n, i] = ele[index] 
				self.phononPerms[n, i] = pho[index] 



	def getPermittivities(self, update=True, anharmonicity=False, anhFreqFiles=[], numberOfFiles=0):
		self.electronPerm = np.zeros((self.ndirs, 3, 3))
		self.phononPerm = np.zeros((self.ndirs, 3, 3))
		self.staticPerm = np.zeros((self.ndirs, 3, 3))
		self.anharmonicity = anharmonicity

		if not anharmonicity:
			anhFreqFiles = [''] * self.ndirs

		if numberOfFiles == 0:
			self.numberLabel = 'all'
		else:
			self.numberLabel = '{}'.format(numberOfFiles)

		for n in range(self.ndirs):
			d = self.dirs[n]
			disp = self.disps[n]
			self.permSampling.addDirectory(d)
			self.permSampling.initialise(disp)
			self.permSampling.average(update=update, save=True, anharmonicity=anharmonicity, 
				anhFreqFile=anhFreqFiles[n], numberOfFiles=numberOfFiles)

			self.electronPerm[n] = self.permSampling.electronPerm
			self.phononPerm[n] = self.permSampling.phononPerm
			self.staticPerm[n] = self.permSampling.staticPerm


	def saveSummary(self, terminal):
		if self.anharmonicity:
			filename = 'permittivity_anh_{}.summary'.format(self.numberLabel)
		else:
			filename = 'permittivity_har_{}.summary'.format(self.numberLabel)
		f = open(join(terminal, filename), 'w')
		f.write('Anharmonicity:	{} \n'.format(self.anharmonicity))
		f.write('Files averaged over: {} \n \n'.format(self.numberLabel))

		for (i, label) in enumerate(self.labels):
			f.write(' ===================== 龴ↀ◡ↀ龴 ===================== \n ')
			f.write('Label:	{} \n \n'.format(label))
			f.write('Electron Contribution: \n')
			f.write(np.array2string(self.electronPerm[i], precision=4))
			f.write('\n')
			f.write('Phonon Contribution: \n')
			f.write(np.array2string(self.phononPerm[i], precision=4))
			f.write('\n')
			f.write('Permittivity: \n')
			f.write(np.array2string(self.staticPerm[i], precision=4))
			f.write('\n')
			eigenvalues = np.linalg.eigvals(self.staticPerm[i])
			f.write(np.array2string(eigenvalues))
			f.write('\n')
			f.write('Geometrical Average: {}'.format(np.prod(eigenvalues) ** (1/3)))
			f.write('\n')
		f.close()

	@classmethod
	def diffSummary(cls, tool1, tool2, terminal):
		'''
		compute the difference between tool1 and tool2, provided that they have the same number of dirs, 
		and save to terminal on the file system
		'''
		assert np.all(tool1.staticPerm.shape == tool2.staticPerm.shape)
		assert tool1.anharmonicity == tool2.anharmonicity
		electronPermDiff = tool1.electronPerm - tool2.electronPerm
		phononPermDiff = tool1.phononPerm - tool2.phononPerm 
		staticPermDiff = tool1.staticPerm - tool2.staticPerm
		anharmonicity = tool1.anharmonicity

		if anharmonicity:
			filename = 'permittivity_diff_anh.summary'
		else:
			filename = 'permittivity_diff_har.summary'
		f = open(join(terminal, filename), 'w')
		f.write('Anharmonicity:	{} \n \n'.format(anharmonicity))
		for (i, label) in enumerate(tool1.labels):
			f.write(' ===================== 龴ↀ◡ↀ龴 ===================== \n ')
			f.write('Label:	{} \n \n'.format(label))
			f.write('Difference in Electron Contribution: \n')
			f.write(np.array2string(electronPermDiff[i], precision=4))
			f.write('\n')
			f.write('Difference in Phonon Contribution: \n')
			f.write(np.array2string(phononPermDiff[i], precision=4))
			f.write('\n')
			f.write('Difference in Permittivity: \n')
			f.write(np.array2string(staticPermDiff[i], precision=4))
			f.write('\n')
			eigenvalues = np.linalg.eigvals(staticPermDiff[i])
			f.write(np.array2string(eigenvalues))
			f.write('\n')
			f.write('Geometrical Average of difference: {}'.format(np.prod(eigenvalues) ** (1/3)))
			f.write('\n')
		f.close()






if __name__ == '__main__':

	'''
	# ======================
	# Temperature Dependence
	# ======================
	temperatures = ['static', '0K', '100K', '240K']
	dirs = [join('Efield/output', x) for x in temperatures]
	dispFile = [join('Efield/input_files', 'disp_patterns.dat')] * len(dirs)
	anhFile = [join('Efield/input_files', 'anharmonic_eigenvalues.dat')] * len(dirs)
	terminal = 'Efield/output'

	tool = PermAnalysisTool(dirs, dispFile)
	tool.getPermittivities(update=True, show=True, anharmonicity=True, anhFreqFile=anhFile)
	tool.saveSummary(terminal)
	''' 


	'''
	# =============
	# Proton Orders  
	# =============

	protonorders = ['IceIh_C1c1', 'IceIh_Cmc21', 'IceIh_Pna21']
	temperatures = ['static', '0K']
	dispFiles = [join('../Efield', protonorder, 'input_files', 'disp_patterns.dat') 
				for protonorder in protonorders]
	anhFiles = [join('../Efield', protonorder, 'anharmonic_eigen', 'anharmonic_eigenvalues.dat') 
				for protonorder in protonorders]
	dir_static = [join('../Efield', protonorder, 'output', temperatures[0])
			for protonorder in protonorders]
	dir_0K = [join('../Efield', protonorder, 'output', temperatures[1])
			for protonorder in protonorders]

	terminal = '../Efield/summaries'

	tool1 = PermAnalysisTool(dir_static, dispFiles)
	tool1.getPermittivities(update=True, anharmonicity=True, anhFreqFiles=anhFiles)
	tool1.saveSummary(join(terminal, temperatures[0]))

	tool2 = PermAnalysisTool(dir_0K, dispFiles)
	tool2.getPermittivities(update=True, anharmonicity=True, anhFreqFiles=anhFiles)
	tool2.saveSummary(join(terminal, temperatures[1]))

	PermAnalysisTool.diffSummary(tool1, tool2, terminal)

	''' 

	# =========== 
	# Functionals 
	# =========== 

	functionals = ['LDA', 'PBE', 'PBE_TS']
	temperatures = ['static', '0K']
	dispFiles = [join('../Functionals', 'LDA', 'input_files', 'disp_patterns.dat')] * len(functionals)
	anhFiles = [join('../Functionals', 'LDA', 'anharmonic_eigen', 'anharmonic_eigenvalues.dat')] * len(functionals)
	dir_static = [join('../Functionals', functional, 'output', temperatures[0]) for functional in functionals]
	dir_0K = [join('../Functionals', functional, 'output', temperatures[1]) for functional in functionals]

	terminal = '../Functionals/summaries'

	# For ground state energy
	tool1 = PermAnalysisTool(dir_static, dispFiles)
	tool1.getPermittivities(update=True, anharmonicity=True, anhFreqFiles=anhFiles, numberOfFiles=0)
	tool1.saveSummary(join(terminal, temperatures[0]))

	# For 0k 
	tool2 = PermAnalysisTool(dir_0K, dispFiles)

	number = 1 
	tool2.getPermittivities(update=True, anharmonicity=True, anhFreqFiles=anhFiles, numberOfFiles=number)
	tool2.saveSummary(join(terminal, temperatures[1]))

	number = 2
	tool2.getPermittivities(update=True, anharmonicity=True, anhFreqFiles=anhFiles, numberOfFiles=number)
	tool2.saveSummary(join(terminal, temperatures[1]))

	number = 10
	for i in range(3):
		tool2.plotSamplePerms(number, terminal, anharmonicity=True, anhFreqFiles=anhFiles, index=(i, i)) # plot the diagonal elements

	number = 20
	tool2.getPermittivities(update=False, anharmonicity=True, anhFreqFiles=anhFiles, numberOfFiles=number)
	tool2.saveSummary(join(terminal, temperatures[1]))

	# Compare the difference between ground state and 0K 
	PermAnalysisTool.diffSummary(tool1, tool2, terminal)










	
























