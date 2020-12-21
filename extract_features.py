import random

import timm
import torch
import cv2
import numpy as np
import pickle
import pathlib

from collections import defaultdict
from utils.utils import casia_enrollment_split, load_train_test
from cnn_feature_extraction.feature_extraction import extract_features_CNN
from tqdm import tqdm
from sklearn.decomposition import PCA

def load_img_to_tensor(filename):
    norm_img = cv2.imread(filename) / 255
    norm_mask = (cv2.imread(filename.replace(".png","_mask.png"), cv2.IMREAD_GRAYSCALE) / 255).astype(bool)
    norm_img[norm_mask] = 0
    return convert_img_to_tensor(norm_img)

def convert_img_to_tensor(mat):
    # mat = np.array(mat)
    mat = Normalize()(mat) # Important!
    height, width, channels = mat.shape
    norm_img_reshaped = mat.reshape((channels, width, height)).astype(np.float32)
    img_tensor = torch.from_numpy(norm_img_reshaped)
    return img_tensor[np.newaxis, :].cuda()

def convert_norm_iris_to_tensor(mat):
    # mat = np.array(mat)
    norm_img_stacked = np.stack((mat,)*3, axis=-1)
    height, width, channels = norm_img_stacked.shape
    norm_img_reshaped = norm_img_stacked.reshape(channels, width, height).astype(np.float32)
    img_tensor = torch.from_numpy(norm_img_reshaped)
    return img_tensor[np.newaxis, :].cuda()

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img -= self.mean
        img /= self.std
        return img


if __name__ == '__main__':
    # extractFeatureCNN("./data/CASIA-Iris-Thousand/000/L/S5000L00.jpg")
    # extractFeature("./data/CASIA1/2/002_1_1.jpg")
    # print(torch.cuda.current_device())
    # print(torch.cuda.get_device_name(0))
    # print( torch.cuda.is_available())
    # model_names = timm.list_models("eff*",pretrained=True)

    dataset_dir = "./data/CASIA_thousand_norm_512_64_e"

    # a2_norm = load_img_to_tensor(f"./data/{dataset}/75/075_1_1.png")
    # a1_norm = load_img_to_tensor(f"./data/{dataset}/75/075_1_2.png")
    # b_norm = load_img_to_tensor(f"./data/{dataset}/6/006_1_1.png")

    # - poskusi z normalizacijo slik
    # - min max scaling

    # model_name  = "densenetblur121d"
    # model_name = "efficientnet_b3a"
    model_name = "resnet50d"
    layer = "layer_act1"
    # model_name = "densenetblur121d"

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.cuda()

    print("Loading dataset...")
    train, test = load_train_test(dataset_dir)


    print("Extracting training features...")
    train_features = defaultdict(list)
    for key in tqdm(train.keys()):
        for filename in train[key][:3]:
            features = extract_features_CNN(filename, model)
            train_features[str(key)].append(features.numpy())

    print("Extracting test features...")
    test_features = defaultdict(list)
    for key in tqdm(test.keys()):
        for filename in test[key][:1]:
            features = extract_features_CNN(filename, model)
            test_features[key].append(features.numpy())

    pathlib.Path(f"./templates/{model_name}/{layer}").mkdir(parents=True, exist_ok=True)

    with open(f"./templates/{model_name}/{layer}/enrolled_features.pickle", "wb") as f:
        pickle.dump(train_features, f, pickle.HIGHEST_PROTOCOL)

    with open(f"./templates/{model_name}/{layer}/test_features.pickle", "wb") as f:
        pickle.dump(test_features, f, pickle.HIGHEST_PROTOCOL)
    #
    # with open(f"./templates/{model_name}/{layer}/enroll_identities.pickle", "wb") as f:
    #     pickle.dump(enrollment, f, pickle.HIGHEST_PROTOCOL)

    # with open(f"./templates/{model_name}/{layer}/test_identities.pickle", "wb") as f:
    #     pickle.dump(test, f, pickle.HIGHEST_PROTOCOL)
