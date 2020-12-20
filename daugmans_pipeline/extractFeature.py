##-----------------------------------------------------------------------------
##  Import
##-----------------------------------------------------------------------------
from cv2 import imread, imwrite

from functions.segment import segment
from functions.normalize import normalize
from daugmans_pipeline.encode import encode


##-----------------------------------------------------------------------------
##  Parameters for extracting feature
##	(The following parameters are default for CASIA1 dataset)
##-----------------------------------------------------------------------------
# Segmentation parameters
eyelashes_thres = 80

# Normalisation parameters
radial_res = 64
angular_res = 256

# Feature encoding parameters
minWaveLength = 18
mult = 1
sigmaOnf = 0.5


##-----------------------------------------------------------------------------
##  Function
##-----------------------------------------------------------------------------
def extractFeature(im_filename, eyelashes_thres=80, use_multiprocess=True):
	"""
	Description:
		Extract features from an iris image

	Input:
		im_filename			- The input iris image
		use_multiprocess	- Use multiprocess to run

	Output:
		template			- The extracted template
		mask				- The extracted mask
		im_filename			- The input iris image
	"""
	# Perform segmentation
	im = imread(im_filename, 0)
	ciriris, cirpupil, imwithnoise = segment(im, eyelashes_thres, use_multiprocess)

	# Perform normalization
	polar_array, noise_array = normalize(imwithnoise, ciriris[1], ciriris[0], ciriris[2],
										 cirpupil[1], cirpupil[0], cirpupil[2],
										 radial_res, angular_res)

	imwrite("test.png", polar_array * 255)
	imwrite("test_noise.png", noise_array * 255)
	# Perform feature encoding
	template, mask = encode(polar_array, noise_array, minWaveLength, mult, sigmaOnf)

	# Return
	return template, mask, im_filename