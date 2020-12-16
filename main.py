
from cnn_feature_extraction.feature_extraction import extractFeatureCNN
import timm
import torch
import cv2
import numpy as np

def load_img_to_tensor(filename):
    norm_img = cv2.imread(filename) / 255
    return convert_img_to_tensor(norm_img)

def convert_img_to_tensor(mat):
    mat = np.array(mat)
    height, width, channels = mat.shape
    norm_img_reshaped = mat.reshape(channels, width, height).astype(np.float32)
    img_tensor = torch.from_numpy(norm_img_reshaped)
    return img_tensor[np.newaxis, :].cuda()

def convert_norm_iris_to_tensor(mat):
    mat = np.array(mat)
    norm_img_stacked = np.stack((mat,)*3, axis=-1)
    height, width, channels = norm_img_stacked.shape
    norm_img_reshaped = norm_img_stacked.reshape(channels, width, height).astype(np.float32)
    img_tensor = torch.from_numpy(norm_img_reshaped)
    return img_tensor[np.newaxis, :].cuda()



if __name__ == '__main__':
    # extractFeatureCNN("./data/CASIA-Iris-Thousand/000/L/S5000L00.jpg")
    # extractFeatureCNN("./data/CASIA1/2/002_1_1.jpg")
    # print(torch.cuda.current_device())
    # print(torch.cuda.get_device_name(0))
    # print( torch.cuda.is_available())
    model_name  = "efficientnet_b3a"
    # model_names = timm.list_models("eff*",pretrained=True)
    model = timm.create_model(model_name, pretrained=True)
    model.cuda()


    a1_norm, a1_mask = extractFeatureCNN("./data/CASIA1/2/002_1_1.jpg")
    a2_norm, a2_mask = extractFeatureCNN("./data/CASIA1/2/002_1_3.jpg")
    b_norm, b_mask = extractFeatureCNN("./data/CASIA1/51/051_1_1.jpg")

    a_1 = convert_norm_iris_to_tensor(a1_norm)
    a_2 = convert_norm_iris_to_tensor(a2_norm)
    b = convert_norm_iris_to_tensor(b_norm)

    #
    # a_1 = load_img_to_tensor("./data/a_1.jpg")
    # a_2 = load_img_to_tensor("./data/a_1_similar.jpg")
    # b = load_img_to_tensor("./data/b_1_black.jpg")
    #
    a_1_features = model.forward_features(a_1).flatten()
    a_2_features = model.forward_features(a_2).flatten()
    b_features = model.forward_features(b).flatten()
    #
    dim = 0
    similarity = torch.nn.CosineSimilarity(dim=dim, eps=1e-6)
    print("similar: ", similarity(a_1_features, a_2_features))
    print("different 1: ", similarity(a_1_features, b_features))
    print("different 2: ", similarity(a_2_features, b_features))

