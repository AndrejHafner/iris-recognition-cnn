import math
import numpy as np

def IrisNormalization(image, inner_circle, outer_circle, row = 64, col = 512):
    localized_img = image
    normalized_iris = np.zeros(shape=(row, col))
    angle = 2.0 * math.pi / col
    inner_boundary_x = np.zeros(shape=(1, col))
    inner_boundary_y = np.zeros(shape=(1, col))
    outer_boundary_x = np.zeros(shape=(1, col))
    outer_boundary_y = np.zeros(shape=(1, col))
    for j in range(col):
        inner_boundary_x[0][j] = inner_circle[0] + inner_circle[2] * math.cos(angle * (j))
        inner_boundary_y[0][j] = inner_circle[1] + inner_circle[2] * math.sin(angle * (j))

        outer_boundary_x[0][j] = outer_circle[0] + outer_circle[2] * math.cos(angle * (j))
        outer_boundary_y[0][j] = outer_circle[1] + outer_circle[2] * math.sin(angle * (j))

    for j in range(512):
        for i in range(64):
            normalized_iris[i][j] = localized_img[min(int(int(inner_boundary_y[0][j])
                                                          + (int(outer_boundary_y[0][j]) - int(
                inner_boundary_y[0][j])) * (i / 64.0)), localized_img.shape[0] - 1)][min(int(int(inner_boundary_x[0][j])
                                                                                             + (int(
                outer_boundary_x[0][j]) - int(inner_boundary_x[0][j]))
                                                                                             * (i / 64.0)),
                                                                                         localized_img.shape[1] - 1)]

    res_image = 255 - normalized_iris
    return res_image


