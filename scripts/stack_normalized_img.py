from PIL import Image
import numpy as np

from utils.utils import get_files_walk

if __name__ == '__main__':
    for file in get_files_walk("./CASIA_thousand_norm_256_64_e_nn_open_set_stacked"):
        img = np.array(Image.open(file))
        new_img = np.zeros((256, 256))
        new_img[0:64, :] = img
        new_img[64:128, :] = img
        new_img[128:192, :] = img
        new_img[192:256, :] = img

        Image.fromarray(new_img).convert("RGB").save(file)