import os
import pathlib
import shutil

from tqdm import tqdm

from utils.utils import parse_casia_thousand_filename

# Specify the directory where you want to transform the dataset
base_dir = "./CASIA_thousand_norm_256_64_e_nn_open_set"

train_dir = os.path.join(base_dir, "enrollment")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")


print("Preparing train dataset for torch.datasets.ImageFolder...")
# Train set
for file in tqdm(os.listdir(train_dir)):
    if os.path.isdir(os.path.join(train_dir,file)):
        continue

    full_file_path = os.path.join(train_dir, file)
    if "_mask" in file:
        os.remove(full_file_path)
        continue

    identifier, side, index = parse_casia_thousand_filename(file)
    folder = os.path.join(train_dir, f"{identifier}_{side}")
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    shutil.move(os.path.abspath(full_file_path), os.path.abspath(os.path.join(folder, file)))

print("Preparing validation dataset for torch.datasets.ImageFolder...")
# Validation set
for file in tqdm(os.listdir(val_dir)):
    if os.path.isdir(os.path.join(val_dir,file)):
        continue

    full_file_path = os.path.join(val_dir, file)
    if "_mask" in file:
        os.remove(full_file_path)
        continue

    identifier, side, index = parse_casia_thousand_filename(file)
    folder = os.path.join(val_dir, f"{identifier}_{side}")
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    shutil.move(os.path.abspath(full_file_path), os.path.abspath(os.path.join(folder, file)))
#

print("Preparing test dataset for torch.datasets.ImageFolder...")
# Validation set
for file in tqdm(os.listdir(test_dir)):
    if os.path.isdir(os.path.join(test_dir,file)):
        continue

    full_file_path = os.path.join(test_dir, file)
    if "_mask" in file:
        os.remove(full_file_path)
        continue

    identifier, side, index = parse_casia_thousand_filename(file)
    folder = os.path.join(test_dir, f"{identifier}_{side}")
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    shutil.move(os.path.abspath(full_file_path), os.path.abspath(os.path.join(folder, file)))



