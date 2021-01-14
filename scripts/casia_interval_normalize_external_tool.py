import itertools
import os
import pathlib
import subprocess

from multiprocessing import Pool
from tqdm import tqdm
from utils.utils import casia_train_val_test_split, parse_casia_interval_filename

dir = "../data/CASIA-Iris-Interval"
target_dir = "../data/CASIA_interval_norm_512_64_e"
width = 512
height = 64
enhancement = True
quiet_mode = True

def pool_func_train(file_path):
    filename = file_path.split("\\")[-1]
    dest_file_path = os.path.join(target_dir, "train", filename).replace(".jpg", ".png")
    command = f"./iris_segm_norm.exe -i {file_path} -o {dest_file_path} -s {width} {height} -m {dest_file_path.replace('.png','_mask.png')} {'-e' if enhancement else ''} {'-q' if quiet_mode else ''}"
    child = subprocess.Popen(command)
    child.communicate()
    return child.returncode, command # Return the code to check if it succedded

def pool_func_test(file_path):
    filename = file_path.split("\\")[-1]
    dest_file_path = os.path.join(target_dir, "test", filename).replace(".jpg", ".png")
    command = f"./iris_segm_norm.exe -i {file_path} -o {dest_file_path} -s {width} {height} -m {dest_file_path.replace('.png','_mask.png')} {'-e' if enhancement else ''} {'-q' if quiet_mode else ''}"
    child = subprocess.Popen(command)
    child.communicate()
    return child.returncode, command # Return the code to check if it succedded

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
