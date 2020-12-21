from collections import defaultdict

import timm
import torch
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

import pickle

from tqdm import tqdm

from cnn_feature_extraction.feature_extraction import extract_features_CNN
from utils.utils import load_train_test

if __name__ == '__main__':

    model_name = "resnet50d"
    layer = "layer_act1"
    dataset_dir = "./data/CASIA_thousand_norm_512_64_e"
    random_state = 42

    print(f"Loading model {model_name}...")
    model = timm.create_model(model_name, pretrained=True)
    model.cuda()

    print("Loading dataset...")
    train, test = load_train_test(dataset_dir)

    train_inter = {}
    test_inter = {}
    classes = 100
    for key in list(train.keys())[:classes]:
        train_inter[key] = train[key]
        test_inter[key] = test[key]

    train = train_inter
    test = test_inter


    print("Extracting training features...")
    train_features = defaultdict(list)
    for key in tqdm(train.keys()):
        for filename in train[key]:
            features = extract_features_CNN(filename, model)
            train_features[str(key)].append(features.numpy())

    print("Extracting test features...")
    test_features = defaultdict(list)
    for key in tqdm(test.keys()):
        for filename in test[key]:
            features = extract_features_CNN(filename, model)
            test_features[key].append(features.numpy())

    train_keys = list(train_features.keys())

    train_X = []
    train_Y = []
    for key in train_keys:
        for val in train_features[key]:
            train_X.append(val)
            train_Y.append(train_keys.index(key))
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)




    test_keys = list(test_features.keys())

    test_X = []
    test_Y = []
    for key in test_keys:
        for val in test_features[key]:
            test_X.append(val)
            test_Y.append(train_keys.index(str(key)))

    test_X = np.array(test_X)
    test_Y = np.array(test_Y)


    pca_n_components = 500
    pipeline = make_pipeline(PCA(n_components=pca_n_components, whiten=True, random_state=random_state),
                             SVC(kernel='rbf', C=5, gamma=0.01))


    print("Training classifier...")
    pipeline.fit(train_X, train_Y)
    print("Scoring classifier...")
    print(pipeline.score(train_X, train_Y))
