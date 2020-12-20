from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

import pickle

from cnn_feature_extraction.feature_extraction import extract_features_CNN

if __name__ == '__main__':

    model_name = "resnet50d"
    layer = "layer_maxpool"


    print("Loading data...")

    # print("Initializing model...")
    # model = timm.create_model(model_name, pretrained=True)
    # model.cuda()

    # Load enrolled identites features
    with open(f"./templates/{model_name}/{layer}/enrolled_features.pickle", "rb") as f:
        enrolled_features = pickle.load(f)

    # Load test identites
    with open(f"./templates/{model_name}/{layer}/test_features.pickle", "rb") as f:
        test_features = pickle.load(f)

    enrolled_keys = list(enrolled_features.keys())

    train_X = []
    train_Y = []
    for key in enrolled_keys:
        for val in enrolled_features[key]:
            train_X.append(val)
            train_Y.append(enrolled_keys.index(key))
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)


    print("Running PCA...")
    pca = PCA(n_components=0.9, svd_solver="full")
    pca.fit(train_X)
    train_X_reduced = pca.transform(train_X)



    test_keys = list(test_features.keys())

    test_X = []
    test_Y = []
    for key in test_keys:
        for val in test_features[key]:
            test_X.append(val)
            test_Y.append(enrolled_keys.index(str(key)))

    test_X = np.array(test_X)
    test_Y = np.array(test_Y)
    test_X_reduced = pca.transform(test_X)



    pipeline = make_pipeline(StandardScaler(), SVC())


    print("Training classifier...")
    pipeline.fit(train_X, train_Y)
    print("Scoring classifier...")
    print(pipeline.score(test_X, test_Y))
