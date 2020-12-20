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
    filename = filename.replace(".png", "")
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


def casia_interval_enrollment_split(dir, split = 0.7, random_seed = 42):
    identities = defaultdict(list)
    enrollment = defaultdict(list)
    test = defaultdict(list)

    for file in get_files_walk(dir):
        if "_mask" in file: continue
        identifier, side, index = parse_casia_interval_filename(file)
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

