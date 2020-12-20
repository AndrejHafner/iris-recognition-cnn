import pathlib
import os
from multiprocessing import Pool

from tqdm import tqdm
from cv2 import imread, imwrite
from common.segment import segment
from common.normalize import normalize


# Segmentation parameters
eyelashes_thresh = 80

# Normalisation parameters
radial_res = 64
angular_res = 256

iris_folder = "../data/CASIA-Iris-Thousand"
normalized_folder = "../data/CASIA_iris_thousand_normalized"


def pool_func(path_tuple):
    filename, orig_folder, dest_folder = path_tuple
    im = imread(os.path.join(orig_folder, filename), 0)
    ciriris, cirpupil, imwithnoise = segment(im, eyelashes_thresh, False)

    # Perform normalization
    polar_iris, polar_mask = normalize(imwithnoise, ciriris[1], ciriris[0], ciriris[2],
                                       cirpupil[1], cirpupil[0], cirpupil[2],
                                       radial_res, angular_res)
    imwrite(os.path.join(dest_folder, filename.replace(".jpg", ".png")), polar_iris * 255)
    imwrite(os.path.join(dest_folder, filename.replace(".jpg", "") + "_mask.png"), polar_mask * 255)


if __name__ == '__main__':
    files = []
    for dirpath, dirnames, filenames in os.walk(iris_folder):
        new_dir = dirpath.replace(iris_folder, normalized_folder)
        pathlib.Path(new_dir).mkdir(parents=True, exist_ok=True)
        files += [(file, dirpath, new_dir) for file in filenames]

    print("Started extracting normalized iris regions...")
    pools = Pool(processes=8)
    for _ in tqdm(pools.imap_unordered(pool_func, files), total=len(files)):
        pass



