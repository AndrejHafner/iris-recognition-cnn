import random
from collections import defaultdict

import torch
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader

from finetuning.eval_open_set import get_model, enroll_identities, evaluate
from torchvision import datasets, transforms

from utils.utils import get_files_walk, parse_casia_thousand_filename

class IrisDataset(Dataset):

    def __init__(self, identities, transforms):
        self.transforms = transforms
        self.samples = []

        # Convert identities into an iterable
        for index, key in enumerate(identities.keys()):
            self.samples += [(index, val) for val in identities[key]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        label, path = self.samples[index]

        sample = default_loader(path)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample, label


def get_dataloader(identities, input_size, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = IrisDataset(identities, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

if __name__ == '__main__':


    print("Loading model...")
    checkpoint_path = "./densenet_models/densenet201_e_80_lr_0_0001_best.pth"
    model_name = "densenet201"

    # checkpoint_path = "./resnet_models/resnet101_e_80_lr_2e-05_best.pth"
    # model_name = "resnet101"

    enrollment_data_path = "./CASIA_thousand_norm_256_64_e_nn_open_set_stacked/enrollment"
    test_data_path = "./CASIA_thousand_norm_256_64_e_nn_open_set_stacked/test"
    batch_size = 196


    model, input_size = get_model(model_name, checkpoint_path)

    device = torch.device('cuda')
    model.to(device)
    model.eval()


    identities = defaultdict(list)
    for file in get_files_walk("./CASIA_thousand_norm_256_64_e_nn_open_set_stacked"):
        identifier, side, index = parse_casia_thousand_filename(file)
        identities[(identifier, side)].append(file)

    # Shuffle the paths
    for key in identities.keys():
        random.shuffle(identities[key])

    images_per_identity = 10
    folds = 5

    test_images_cnt = int(images_per_identity / folds)
    accuracies = []

    print(f"Starting cross validation, folds: {folds}")
    for fold in range(folds):
        print(f"CV - fold {fold+1}/{folds}")
        # Split into an enrollment and test set
        cv_enrolled = {}
        cv_test = {}
        test_indices = list(range(fold * test_images_cnt, fold * test_images_cnt + test_images_cnt))
        enrollment_indices = list(set(range(images_per_identity)) - set(test_indices))

        for key in identities.keys():
            identity = identities[key]
            cv_enrolled[key] = [path for i, path in enumerate(identity) if i in enrollment_indices]
            cv_test[key] = [path for i, path in enumerate(identity) if i in test_indices]

        cv_enrollment_dataloader = get_dataloader(cv_enrolled, input_size, batch_size=batch_size)
        cv_test_dataloader = get_dataloader(cv_test, input_size, batch_size=batch_size)

        enrolled = enroll_identities(model.feature_extract_avg_pool, cv_enrollment_dataloader, device)
        accuracy = evaluate(enrolled, model.feature_extract_avg_pool, cv_test_dataloader, device)
        accuracies.append(accuracy)

    print(f"mean CV accuracy: {np.mean(accuracies)}")
