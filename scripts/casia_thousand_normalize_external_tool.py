import itertools
import os
import pathlib
import subprocess

from multiprocessing import Pool
from tqdm import tqdm
from utils.utils import casia_train_val_test_split, parse_casia_thousand_filename

dir = "../data/CASIA-Iris-Thousand"
width = 256
height = 64
enhancement = True
quiet_mode = True
target_dir = f"../data/CASIA_thousand_norm_{width}_{height}{'_e' if enhancement else ''}_nn_open_set"


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

def pool_func_val(file_path):
    filename = file_path.split("\\")[-1]
    dest_file_path = os.path.join(target_dir, "val", filename).replace(".jpg", ".png")
    command = f"./iris_segm_norm.exe -i {file_path} -o {dest_file_path} -s {width} {height} -m {dest_file_path.replace('.png','_mask.png')} {'-e' if enhancement else ''} {'-q' if quiet_mode else ''}"
    child = subprocess.Popen(command)
    child.communicate()
    return child.returncode, command # Return the code to check if it succedded

if __name__ == '__main__':

    train_dict, val_dict, test_dict = casia_train_val_test_split(dir, parse_func=parse_casia_thousand_filename, from_=1500, to=2000)

    train = list(itertools.chain.from_iterable(train_dict.values()))#[: 7*50]
    val = list(itertools.chain.from_iterable(val_dict.values()))#[: 7*50]
    test = list(itertools.chain.from_iterable(test_dict.values()))#[: 3*50]

    pathlib.Path(os.path.join(target_dir, "train")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(target_dir, "val")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(target_dir, "test")).mkdir(parents=True, exist_ok=True)


    print("Started extracting normalized iris regions for training...")
    pools = Pool(processes=8)
    for ret_code, command in tqdm(pools.imap_unordered(pool_func_train, train), total=len(train)):
        if ret_code != 0:
            print(f"Error extracting iris: {command}")

    print("Started extracting normalized iris regions for validation...")
    pools = Pool(processes=8)
    for ret_code, command in tqdm(pools.imap_unordered(pool_func_val, val), total=len(val)):
        if ret_code != 0:
            print(f"Error extracting iris: {command}")


    print("Started extracting normalized iris regions for testing...")
    pools = Pool(processes=8)
    for ret_code, command in tqdm(pools.imap_unordered(pool_func_test, test), total=len(test)):
        if ret_code != 0:
            print(f"Error extracting iris: {command}")
