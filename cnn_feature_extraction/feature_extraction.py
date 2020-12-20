import cv2
import numpy as np
import torch
import timm

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

def mask_out_region(filename, image):
    norm_mask = (cv2.imread(filename.replace(".png","_mask.png"), cv2.IMREAD_GRAYSCALE) / 255).astype(bool)
    image[norm_mask] = 0
    return image

def convert_img_to_tensor(mat, cuda = True):
    # mat = np.array(mat)
    mat = Normalize()(mat) # Important!
    height, width, channels = mat.shape
    norm_img_reshaped = mat.reshape((channels, width, height)).astype(np.float32)
    img_tensor = torch.from_numpy(norm_img_reshaped)
    if cuda:
        return img_tensor[np.newaxis, :].cuda()
    return img_tensor[np.newaxis, :]

def convert_norm_iris_to_tensor(mat):
    # mat = np.array(mat)
    norm_img_stacked = np.stack((mat,)*3, axis=-1)
    height, width, channels = norm_img_stacked.shape
    norm_img_reshaped = norm_img_stacked.reshape(channels, width, height).astype(np.float32)
    img_tensor = torch.from_numpy(norm_img_reshaped)
    return img_tensor[np.newaxis, :].cuda()


@torch.no_grad()
def extract_features_CNN(filename, model, cuda = True):
    norm_img = cv2.imread(filename) / 255
    norm_img_masked = mask_out_region(filename, norm_img)
    img_tensor = convert_img_to_tensor(norm_img_masked, cuda = cuda)
    features =  model.forward_features(img_tensor).flatten().cpu()
    return features