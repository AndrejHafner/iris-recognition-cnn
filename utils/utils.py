import os
import random
from collections import defaultdict

def parse_casia_interval_filename(filename):
    """

    :param filename: Filename of the Casia-Iris-Interval image
    :return:
    identifier -> unique ID of the subject
    side -> L or R for left or right eye - unique per person
    index -> index of the image for this class - there are multiple pictures per eye
    """
    filename = filename.replace(".png", "").replace(".jpg", "")
    index = int(filename[-2:])
    side = filename[-3:-2]
    identifier = int(filename[-6:-3])
    return identifier, side, index

def parse_casia_thousand_filename(filename):
    """

    :param filename: Filename of the Casia-Iris-Interval image
    :return:
    identifier -> unique ID of the subject
    side -> L or R for left or right eye - unique per person
    index -> index of the image for this class - there are multiple pictures per eye
    """
    filename = filename.replace(".jpg", "").replace(".png", "")
    index = int(filename[-2:])
    side = filename[-3:-2]
    identifier = int(filename[-6:-3])
    return identifier, side, index



def get_files_walk(dir):
    """
    Generate all filenames in a given dir
    :param dir:
    :return:
    """
    for dirpath, dirnames, filenames in os.walk(dir):
        for file in filenames:
            yield os.path.join(dirpath, file)


def casia_enrollment_split(dir, parse_func=parse_casia_interval_filename, split = 0.7, random_seed = 42):
    identities = defaultdict(list)
    enrollment = defaultdict(list)
    test = defaultdict(list)

    for file in get_files_walk(dir):
        if "_mask" in file: continue
        identifier, side, index = parse_func(file)
        identities[(identifier, side)].append(file)

    random.seed(random_seed)
    # Filter out identities with only a single picture of iris -> can't do recognition with only a single image per class??
    identities_filtered = { key : random.sample(sorted(identities[key], key=lambda x: x[0]), len(identities[key])) for key in identities.keys() if len(identities[key]) >= 2 }

    # Split identites to enrollment and testing set
    for key in identities_filtered.keys():
        split_idx = round(len(identities_filtered[key]) * split)
        enrollment[key] = identities_filtered[key][:split_idx]
        test[key] = identities_filtered[key][split_idx:]

    return enrollment, test

def load_train_test(dir, filename_parse_func=parse_casia_thousand_filename):
    train_files = [os.path.join(dir, "train", file) for file in os.listdir(os.path.join(dir, "train")) if "_mask" not in file]
    test_files = [os.path.join(dir, "test", file) for file in os.listdir(os.path.join(dir, "test")) if "_mask" not in file]

    train_dict = defaultdict(list)
    for file in train_files:
        identifier, side, _ = filename_parse_func(file)
        train_dict[(identifier, side)].append(file)

    test_dict = defaultdict(list)
    for file in test_files:
        identifier, side, _ = filename_parse_func(file)
        test_dict[(identifier, side)].append(file)

    return train_dict, test_dict
