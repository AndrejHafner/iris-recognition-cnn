import os
from cv2 import imread
from functions.segment import segment
from functions.normalize import normalize

# Segmentation parameters
eyelashes_thres = 80

# Normalisation parameters
radial_res = 20
angular_res = 240

# Feature encoding parameters
minWaveLength = 18
mult = 1
sigmaOnf = 0.5

im = imread(im_filename, 0)
ciriris, cirpupil, imwithnoise = segment(im, eyelashes_thres, True)

# Perform normalization
polar_iris, polar_mask = normalize(imwithnoise, ciriris[1], ciriris[0], ciriris[2],
                 cirpupil[1], cirpupil[0], cirpupil[2],
                 radial_res, angular_res)