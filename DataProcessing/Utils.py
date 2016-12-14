import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
from scipy import stats, integrate

def fracs2Coordinates(fracs, baseVectors, shape):
	return np.dot(fracs, baseVectors).reshape((*shape, 3), order = 'F')

def shape2Coordinates(shape, baseVectors, tile, shift = False):
	# generate a grid of indices
	x, y, z = [(np.arange(l) - int(l/2)) for l in shape]
	indices = np.zeros((*shape, 3))
	indices[..., 0], indices[..., 1], indices[..., 2] = np.meshgrid(x, y, z)
	# rescale the grid to be the coordinates
	baseVectors = (baseVectors.T/tile).T
	return np.dot(indices, baseVectors)

def fft(realDensities, outshape = (10, 10, 10), tile = (1, 1, 1)):
	# Tile the realDensities across a few unit cells 
	extendedDensities = np.tile(realDensities, tile)
	shape = np.array(extendedDensities.shape)
	assert np.all(shape == realDensities.shape * np.array(tile))
	# Perform fourier transform and shift to have (0, 0, 0) at the centre
	axes = (0, 1, 2)
	reciDensities = np.fft.fftn(extendedDensities, s = shape, axes = axes, norm = None)
	reciDensities = np.fft.fftshift(reciDensities)
	# Crop the reciDensities to only obtain the outshape needed 
	mid = (shape/2).astype('int')
	ranges = [] 
	for i in range(3):
		m = mid[i]
		x = outshape[i]
		r = int(x/2)
		if x % 2 != 0:
			ranges.append(slice(m - r, m + r + 1, 1))
		else:
			ranges.append(slice(m - r, m + r, 1))
	return reciDensities[ranges[0], ranges[1], ranges[2]]

def getReciprocalLatticeVector(realLatticeVectors):
	v0 = realLatticeVectors[0] 
	v1 = realLatticeVectors[1]
	v2 = realLatticeVectors[2]
	# Compute the cross products
	c0 = np.cross(v1, v2)
	c1 = np.cross(v2, v0)
	c2 = np.cross(v0, v1)
	# Compute the volume
	v = np.dot(v0, c0)
	# Compute the reciprocal lattice vectors
	reciprocalLatticeVectors = np.zeros(realLatticeVectors.shape)
	reciprocalLatticeVectors[0] = 2 * np.pi * c0/v
	reciprocalLatticeVectors[1] = 2 * np.pi * c1/v
	reciprocalLatticeVectors[2] = 2 * np.pi * c2/v
	return reciprocalLatticeVectors

def plotAlongCAxis(densities, coordinates, figureName, title = ' ', sum = False, index = 0, contour = True, nlines = 20):
	if sum:
		densitySlice = np.sum(densities, -1)
	else:
		densitySlice = densities[:, :, index]
	if contour:
		plt.contourf(coordinates[:, :, index, 0], coordinates[:, :, index, 1], densitySlice, nlines, cmap = 'plasma')
		plt.axes().set_aspect('equal')
		plt.colorbar()
		plt.title(title)
		plt.savefig('{}.eps'.format(figureName))
		plt.close()
	else:
		plt.scatter(coordinates[:, :, index, 0], coordinates[:, :, index, 1], c = densitySlice, marker = 'H', cmap = 'YlGnBu', alpha = 0.4, s = 12, linewidth = 0)
		plt.axes().set_aspect('equal')
		plt.colorbar()
		plt.title(title)
		plt.savefig('{}.eps'.format(figureName))
		plt.close()

def frac2real(fracPositions, latticeVectors):
	return np.dot(fracPositions, latticeVectors)

def real2frac(realPositions, latticeVectors):
	inverseLatticeVectors = np.linalg.inv(latticeVectors)
	return np.dot(realPositions, inverseLatticeVectors)

def getAnharmonicityLabel(anharmonicity):
	if anharmonicity:
		return 'anh'
	else:
		return 'har'

def calculateTotalNumberOfOrbitals(nMax, lMax):
	'''
	Calculates the maximum number of orbitals from the maximum principle quantum number (nMax)
	and the maximum angular momentum quantum number (lMax)
	'''
	maxOrbitalNumber = nMax * (nMax + 1) / 2
	return (maxOrbitalNumber - (nMax - lMax - 1))

def plotOrbitals(r, wavefunctions):
	shape = wavefunctions.shape
	assert shape[-1] == r.size 

	for i in range(shape[0]):
		plt.plot(r, wavefunctions[i])
	plt.xscale('log')
	plt.show()

def integrateRadialWavefunction(radius, radialWavefunction):
	'''
	radius: radius in logarithmic grid
	radialWavefunction: the corresponding wavefunction
	'''
	integrand = np.einsum('j, j -> j', radialWavefunction ** 2, radius) # r * (r * phi)^2 for each orbital
	logR = np.zeros(radius.shape)
	logR[1:] = np.log(radius[1:]) # so it is eventually spaced 
	return integrate.simps(integrand, logR, axis = -1) 

def plotCellFrameInABPlane(latticeVectors):
	p1 = [0, 0, 0] 
	p2 = latticeVectors[0] 
	p3 = latticeVectors[1] 
	p4 = latticeVectors[0] + latticeVectors[1]
	mins = np.min([p1, p2, p3, p4], axis = 0)
	maxs = np.max([p1, p2, p3, p4], axis = 0)

	plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b--')
	plt.plot([p1[0], p3[0]], [p1[1], p3[1]], 'b--')
	plt.plot([p2[0], p4[0]], [p2[1], p4[1]], 'b--')
	plt.plot([p3[0], p4[0]], [p3[1], p4[1]], 'b--')
	plt.xlim([mins[0], maxs[0]])
	plt.ylim([mins[1], maxs[1]])
	plt.axes().set_aspect('equal')

def denfmt2cell(denFile):
	'''
	Produce the corresponding .cell filename for a .den_fmt filename
	'''
	seedname, index, _ = denFile.split('.')
	return '.'.join(('-'.join((seedname, 'out')), index, 'cell'))





















