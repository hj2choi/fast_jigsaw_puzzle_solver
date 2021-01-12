import numpy as np
import cv2

#direction ENUM
DIR = {
    'd': 0,
    'u': 1,
    'r': 2,
    'l': 3,
}

"""
compares two images and evaluates whether two borders can be stitched seamlessly

idea 1: use simple RMSE around two image borders
        skip pixels for optimization
        similarity = 1/(1+distance)

idea 2: compute colors around border of both sides.
        use RMSE with weights for distance measure.

idea 3: use Mahalanobis Gradient Compatability (MGC)
[ref] Andrew C Gallagher. Jigsaw puzzles with pieces of unknown orientation. In CVPR 2012
[ref2] Sobel filter

@Parameters
img1 (npArray):     raw image (h, w, RGB_256)
img2 (npArray):     raw image (h, w, RGB_256)
dir (uint2):        stitching direction (down, up, right, left)

@Returns
similarity (float): border similarity score
"""
def img_borders_similarity(img1, img2, dir):
    if dir < 2 and len(img1[0]) != len(img2[0]):
        return -1
    elif len(img1) != len(img2):
        return -1

    if dir == DIR['d']:
        return 1/(1+np.mean(np.linalg.norm(np.subtract(img1[-1], img2[0]), axis = 1)))
    elif dir == DIR['u']:
        return 1/(1+np.mean(np.linalg.norm(np.subtract(img2[-1], img1[0]), axis = 1)))
    elif dir == DIR['r']:
        return 1/(1+np.mean(np.linalg.norm(np.subtract(img1[:,-1], img2[:,0]), axis = 1)))
    else:
        return 1/(1+np.mean(np.linalg.norm(np.subtract(img2[:,-1], img1[:,0]), axis = 1)))

"""
flips image across y axis
@Parameters
img (npArray):      raw image (h, w, RGB)
"""
def mirror(img):
    print("mirror", end = " ")
    img_out = np.copy(img)
    for i in range(len(img_out)):
        for j in range(len(img_out[0])//2):
            temp = np.copy(img_out[i][j])
            img_out[i][j] = img_out[i][-j-1]
            img_out[i][-j-1] = temp
    return img_out

"""
flips image across x axis
@Parameters
img (npArray):      raw image (h, w, RGB)
"""
def flip(img):
    print("flip", end = " ")
    img_out = np.copy(img)
    for j in range(len(img_out[0])):
        for i in range(len(img_out)//2):
            temp = np.copy(img_out[i][j])
            img_out[i][j] = img_out[-i-1][j]
            img_out[-i-1][j] = temp
    return img_out

"""
rotates image by 90 degrees by taking bottom left corner as the center
@Parameters
img (npArray):          raw image (h, w, RGB)
clockwise (bool):       clockwise or counterclockwise
"""
def rotate90(img, clockwise = False):
    print("rotate", end = " ")
    img_rot = np.zeros((len(img[0]), len(img), len(img[0][0])),
                            dtype = np.uint8)
    for i in range(len(img_rot)):
        for j in range(len(img_rot[0])):
            img_rot[i][j] = img[-j-1][i] if clockwise else img[j][-i-1]
    return img_rot
