""" assembler_helpers.py
similarity function for stitching images
and 3D distance matrix index mapper (third axis) for performance optimization
"""

import numpy as np

# direction ENUM
DIR_ENUM = {
    'd': 0,
    'u': 1,
    'r': 2,
    'l': 3,
}


def img_borders_similarity(img1, img2, direction):
    """
        Evaluates stitching score of two images.

        Computes Euclidean distance between each pixels of two image borderlines
        and returns their mean.

        Todo1: skip pixels for optimization
        Todo2: compute colors around border of both sides by applying Sobel filter
        Todo3: use RMSE with weights for distance measure.
        Todo4: use Mahalanobis Gradient Compatability (MGC) Andrew C Gallagher in CVPR 2012

        @Parameters
        img1 (npArray):     raw image (h, w, c)
        img2 (npArray):     raw image (h, w, c)
        dir (uint2):        stitching direction (down, up, right, left)

        @Returns
        similarity (float): border similarity score
    """
    if direction < 2 and len(img1[0]) != len(img2[0]):
        return -1
    if len(img1) != len(img2):
        return -1

    distance = -2
    if direction == DIR_ENUM['d']:
        distance = np.mean(np.linalg.norm(np.subtract(img1[-1], img2[0]), axis=1))
    if direction == DIR_ENUM['u']:
        distance = np.mean(np.linalg.norm(np.subtract(img2[-1], img1[0]), axis=1))
    if direction == DIR_ENUM['r']:
        distance = np.mean(np.linalg.norm(np.subtract(img1[:, -1], img2[:, 0]), axis=1))
    if direction == DIR_ENUM['l']:
        distance = np.mean(np.linalg.norm(np.subtract(img2[:, -1], img1[:, 0]), axis=1))
    return 1/(1+distance)


"""
description of map8(tgt_transform, src_transform):

given a fixed orientation of target image cell,
possible src orientations = rotation x flip x 4directions = 4 * 2 * 4 = 32
             [0~7]
              src
  [16~23] src tgt src [24~31]
              src
             [8~15]
total possible cases = 32 x 8 tgt cell orientations = 32 x 8 = 256
below is the mapping table for [256 possible cases] => [32 cases with fixed tgt orientation]
"""
_MAPPING_TABLE8 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                   27, 24, 25, 26, 31, 28, 29, 30, 19, 16, 17, 18, 23, 20, 21,
                   22, 3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14,
                   10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 26,
                   27, 24, 25, 30, 31, 28, 29, 18, 19, 16, 17, 22, 23, 20, 21,
                   17, 18, 19, 16, 21, 22, 23, 20, 25, 26, 27, 24, 29, 30, 31,
                   28, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0, 5, 6, 7, 4,
                   12, 15, 14, 13, 8, 11, 10, 9, 4, 7, 6, 5, 0, 3, 2, 1, 20,
                   23, 22, 21, 16, 19, 18, 17, 28, 31, 30, 29, 24, 27, 26, 25,
                   29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19,
                   18, 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2,
                   6, 5, 4, 7, 2, 1, 0, 3, 14, 13, 12, 15, 10, 9, 8, 11, 30,
                   29, 28, 31, 26, 25, 24, 27, 22, 21, 20, 23, 18, 17, 16, 19,
                   23, 22, 21, 20, 19, 18, 17, 16, 31, 30, 29, 28, 27, 26, 25,
                   24, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8]


def map8(tgt_transform, src_transform):
    """
    rectangular cells: mapping for [254 cases] = >[32 cases with fixed tgt]
    """
    return _MAPPING_TABLE8[(tgt_transform << 5) + src_transform]


def mat_sym_dmap32(trf):
    """
    rectangular image cells: used for filling out symmetric part
    (lower triangular region) in distance matrix
    """
    return [8, 19, 2, 25, 4, 21, 14, 31, 0, 27, 10, 17, 12, 29, 6, 23,
            24, 11, 18, 1, 28, 5, 22, 15, 16, 3, 26, 9, 20, 13, 30, 7][trf]


def mat_sym_dmap16(trf):
    """
    square image cells: used for filling out symmetric part
    (lower triangular region) in distance matrix
    """
    return [4, 5, 2, 3, 0, 1, 6, 7, 12, 9, 14, 11, 8, 13, 10, 15][trf]


def map4(tgt_transform, src_transform):
    """
    square image cells: mapping for [64 cases] = >[16 cases with fixed tgt]
    """
    return {
        0: src_transform, 1: _flip_x(src_transform),
        2: _flip_y(src_transform), 3: _flip_x(_flip_y(src_transform))
    }[tgt_transform]


def _flip_x(i):
    """
    square image cells: mapping for horizontal flip
    """
    return [1, 0, 3, 2, 5, 4, 7, 6, 13, 12, 15, 14, 9, 8, 11, 10][i]


def _flip_y(i):
    """
    square image cells: mapping for vertical flip
    """
    return [6, 7, 4, 5, 2, 3, 0, 1, 10, 11, 8, 9, 14, 15, 12, 13][i]


"""
    mapping table generation routines.
"""


def _generate_mapping_table8(sim_matrix):
    """
    Automatically build index map for rectangular images. (Only run once)
    """
    mapping_table = []
    for i in range(len(sim_matrix[0][1])):
        flag = 0
        for j in range(32):
            print(i, j, "=>", sim_matrix[0][2][i], sim_matrix[0][2][j], end="")
            if np.abs(sim_matrix[0][2][i] - sim_matrix[0][2][j]) < 0.0000001:
                if not flag:
                    print(" match", end="")
                    flag = 1
                    mapping_table.append(j)
                else:
                    print("\n SOMETHING's WRONG")
                    return
            print()
        print()

    print(mapping_table)


def _generate_mapping_table4(sim_matrix):
    """
    Automatically build index map for square images. (Only run once)
    """
    print(sim_matrix[0][1])
    print(sim_matrix[1][0])
    mapping_table = []
    for i in range(32):
        flag = 0
        for j in range(32):
            print(i, j, "=>", sim_matrix[0][1][i], sim_matrix[1][0][j], end="")
            if np.abs(sim_matrix[0][1][i] - sim_matrix[1][0][j]) < 0.0000001:
                if not flag:
                    print(" match", end="")
                    flag = 1
                    mapping_table.append(j)
                else:
                    print("\n SOMETHING's WRONG")
                    return
            print()
        print()

    print(mapping_table)
