import timm
import torch
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from cnn_feature_extraction.feature_extraction import extractFeatureCNN
from functions.extractFeature import extractFeature


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

    dataset = "CASIA1_normalized"

    a2_norm = load_img_to_tensor(f"./data/{dataset}/75/075_1_1.png")
    a1_norm = load_img_to_tensor(f"./data/{dataset}/75/075_1_2.png")
    b_norm = load_img_to_tensor(f"./data/{dataset}/6/006_1_1.png")

    # - poskusi z normalizacijo slik
    # - min max scaling

    # model_name  = "efficientnet_b3a"
    # model_name = "resnet50d"
    model_name = "densenetblur121d"

    model = timm.create_model(model_name, pretrained=True)
    model.cuda()
    a_1_features = model.forward_features(a1_norm).flatten()
    a_2_features = model.forward_features(a2_norm).flatten()
    b_features = model.forward_features(b_norm).flatten()

    dim = 0
    similarity = torch.nn.CosineSimilarity(dim=dim, eps=1e-6)
    print("similar: ", similarity(a_1_features, a_2_features))
    print("different 1: ", similarity(a_1_features, b_features))
    print("different 2: ", similarity(a_2_features, b_features))


    # scaler = MinMaxScaler()
    #
    #
    # a_1_features = torch.from_numpy(scaler.fit_transform(a_1_features.cpu().detach().reshape(-1,1)))
    # a_2_features = torch.from_numpy(scaler.fit_transform(a_2_features.cpu().detach().reshape(-1,1)))
    # b_features = torch.from_numpy(scaler.fit_transform(b_features.cpu().detach().reshape(-1,1)))