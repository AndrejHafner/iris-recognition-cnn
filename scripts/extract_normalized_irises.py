import pathlib
import os
from multiprocessing import Pool

from tqdm import tqdm
from cv2 import imread, imwrite
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

iris_folder = "../data/CASIA1"
normalized_folder = "../data/CASIA1_normalized/"


def pool_func(path_tuple):
    filename, orig_folder, dest_folder = path_tuple
    im = imread(os.path.join(orig_folder, filename), 0)
    ciriris, cirpupil, imwithnoise = segment(im, eyelashes_thres, False)

    # Perform normalization
    polar_iris, polar_mask = normalize(imwithnoise, ciriris[1], ciriris[0], ciriris[2],
                                       cirpupil[1], cirpupil[0], cirpupil[2],
                                       radial_res, angular_res)
    imwrite(os.path.join(dest_folder, filename.replace(".jpg", ".png")), polar_iris * 255)
    imwrite(os.path.join(dest_folder, filename.replace(".jpg", "") + "_mask.png"), polar_mask * 255)


if __name__ == '__main__':
    files = []

    for folder in os.listdir(iris_folder):
        pathlib.Path(normalized_folder + folder).mkdir(parents=True, exist_ok=True)
        for file in os.listdir(os.path.join(iris_folder, folder)):
            orig_folder = os.path.join(iris_folder, folder)
            dest_folder = os.path.join(normalized_folder, folder)
            files.append((file, orig_folder, dest_folder))

    print("Started extracting normalized iris regions...")
    pools = Pool(processes=8)
    for _ in tqdm(pools.imap_unordered(pool_func, files), total=len(files)):
        pass



