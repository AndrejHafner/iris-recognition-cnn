##-----------------------------------------------------------------------------
##  Import
##-----------------------------------------------------------------------------
import numpy as np


##-----------------------------------------------------------------------------
##  Function
##-----------------------------------------------------------------------------
def encode(polar_array, noise_array, minWaveLength, mult, sigmaOnf):
	"""
	Description:
		Generate iris template and noise mask from the normalised iris region.

	Input:
		polar_array		- Normalised iris region.
		noise_array		- Normalised noise region.
		minWaveLength	- Base wavelength.
		mult			- Multicative factor between each filter.
		sigmaOnf		- Bandwidth parameter.

	Output:
		template		- The binary iris biometric template.
		mask			- The binary iris noise mask.
	"""
	# Convolve normalised region with Gabor filters
	filterbank = gaborconvolve(polar_array, minWaveLength, mult, sigmaOnf)

	length = polar_array.shape[1]
	template = np.zeros([polar_array.shape[0], 2 * length])
	h = np.arange(polar_array.shape[0])

	# Create the iris template
	mask = np.zeros(template.shape)
	eleFilt = filterbank[:, :]

	# Phase quantization
	H1 = np.real(eleFilt) > 0
	H2 = np.imag(eleFilt) > 0

	# If amplitude is close to zero then phase data is not useful,
	# so mark off in the noise mask
	H3 = np.abs(eleFilt) < 0.0001
	for i in range(length):
		ja = 2 * i

		# Construct the biometric template
		template[:, ja] = H1[:, i]
		template[:, ja + 1] = H2[:, i]

		# Create noise mask
		mask[:, ja] = noise_array[:, i] | H3[:, i]
		mask[:, ja + 1] = noise_array[:, i] | H3[:, i]

	# Return
	return template, mask


#------------------------------------------------------------------------------
def gaborconvolve(im, minWaveLength, mult, sigmaOnf):
	"""
	Description:
		Convolve each row of an image with 1D log-Gabor filters.

	Input:
		im   			- The image to be convolved.
		minWaveLength   - Wavelength of the basis filter.
		mult   			- Multiplicative factor between each filter.
		sigmaOnf   		- Ratio of the standard deviation of the
						  Gaussian describing the log Gabor filter's transfer
						  function in the frequency domain to the filter center
						  frequency.

	Output:
		filterbank		- The 1D cell array of complex valued convolution
						  resultsCircle coordinates.
	"""
	# Pre-allocate
	rows, ndata = im.shape					# Size
	logGabor = np.zeros(ndata)				# Log-Gabor
	filterbank = np.zeros([rows, ndata], dtype=complex)

	# Frequency values 0 - 0.5
	radius = np.arange(ndata/2 + 1) / (ndata/2) / 2
	radius[0] = 1

	# Initialize filter wavelength
	wavelength = minWaveLength

	# Calculate the radial filter component
	fo = 1 / wavelength 		# Centre frequency of filter
	logGabor[0 : int(ndata/2) + 1] = np.exp((-(np.log(radius/fo))**2) / (2 * np.log(sigmaOnf)**2))
	logGabor[0] = 0

	# For each row of the input image, do the convolution
	for r in range(rows):
		signal = im[r, 0:ndata]
		imagefft = np.fft.fft(signal)
		filterbank[r , :] = np.fft.ifft(imagefft * logGabor)

	# Return
	return filterbank