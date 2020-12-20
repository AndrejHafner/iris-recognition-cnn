import timm
import pickle

from sklearn.preprocessing import StandardScaler

from cnn_feature_extraction.feature_extraction import extract_features_CNN
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
from ast import literal_eval as make_tuple
from sklearn.decomposition import PCA


def compute_cosine_similarity(identity_features, enrolled_features):
    results = []
    scaler = StandardScaler()
    for key in enrolled_features.keys():
        results += [( key, dot(identity_features, val)/(norm(identity_features)*norm(val))) for val in enrolled_features[key]]
    return results

if __name__ == '__main__':

    model_name = "resnet50d"
    layer = "layer_maxpool"

    print("Loading data...")
    # Load enrolled identites features
    with open(f"./templates/{model_name}/{layer}/enrolled_features.pickle", "rb") as f:
        enrolled_features = pickle.load(f)

    # Load test identites
    with open(f"./templates/{model_name}/{layer}/test_features.pickle", "rb") as f:
        test_features = pickle.load(f)

    print("Calculating similarity...")
    results = {}
    for key in tqdm(test_features.keys()):
        for features in test_features[key]:
            results[key] = compute_cosine_similarity(features, enrolled_features)


    correct = 0
    for key in results.keys():
        closest = max(results[key], key=lambda x: x[1])
        # print(f"key:{key}, closest: {closest}")
        if make_tuple(closest[0]) == key:
            print(f"Correct! key:{key}, closest: {closest}")
            correct +=1


    print(f"Accuracy: {correct / len(results)}")