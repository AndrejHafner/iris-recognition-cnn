import itertools
import os
import pathlib
import subprocess

from multiprocessing import Pool

import cv2
from tqdm import tqdm

from common.Enhancement import ImageEnhancement
from common.Normalization import IrisNormalization
from common.Segmentation import IrisLocalization
from utils.utils import casia_train_val_test_split, parse_casia_interval_filename

dir = "../data/CASIA-Iris-Interval"
target_dir = "../data/CASIA_interval_norm_internal_512_64_e"
width = 512
height = 64
enhancement = True
quiet_mode = True

def pool_func_train(file_path):
    filename = file_path.split("\\")[-1]
    dest_file_path = os.path.join(target_dir, "train", filename).replace(".jpg", ".png")
    img = cv2.imread(file_path, 0)
    iris, pupil = IrisLocalization(img)
    normalized = IrisNormalization(img, pupil, iris)
    ROI = ImageEnhancement(normalized)
    cv2.imwrite(dest_file_path, ROI)

def pool_func_test(file_path):
    filename = file_path.split("\\")[-1]
    dest_file_path = os.path.join(target_dir, "test", filename).replace(".jpg", ".png")
    img = cv2.imread(file_path, 0)
    iris, pupil = IrisLocalization(img)
    normalized = IrisNormalization(img, pupil, iris)
    ROI = ImageEnhancement(normalized)
    cv2.imwrite(dest_file_path, ROI)

if __name__ == '__main__':

    train_dict, test_dict = casia_train_val_test_split(dir, parse_func=parse_casia_interval_filename)

    train = list(itertools.chain.from_iterable(train_dict.values()))
    test = list(itertools.chain.from_iterable(test_dict.values()))

    pathlib.Path(os.path.join(target_dir, "train")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(target_dir, "test")).mkdir(parents=True, exist_ok=True)


    print("Started extracting normalized iris regions for training...")
    pools = Pool(processes=8)
    for ret_code, command in tqdm(pools.imap_unordered(pool_func_train, train), total=len(train)):
        if ret_code != 0:
            print(f"Error extracting iris: {command}")


    print("Started extracting normalized iris regions for testing...")
    pools = Pool(processes=8)
    for ret_code, command in tqdm(pools.imap_unordered(pool_func_test, test), total=len(test)):
        if ret_code != 0:
            print(f"Error extracting iris: {command}")
