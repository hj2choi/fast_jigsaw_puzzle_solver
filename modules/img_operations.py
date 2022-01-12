import numpy as np

#direction ENUM
DIR = {
    'd': 0,
    'u': 1,
    'r': 2,
    'l': 3,
}


def img_borders_similarity(img1, img2, dir):
    """
        compares two images and evaluates whether two borders can be stitched seamlessly

        idea 1: use simple RMSE around two image borders
                skip pixels for optimization
                similarity = 1/(1+distance)
        idea 2: compute colors around border of both sides. by applying Sobel filter
                use RMSE with weights for distance measure.
        idea 3: use Mahalanobis Gradient Compatability (MGC) Andrew C Gallagher in CVPR 2012

        @Parameters
        img1 (npArray):     raw image (h, w, RGB_256)
        img2 (npArray):     raw image (h, w, RGB_256)
        dir (uint2):        stitching direction (down, up, right, left)

        @Returns
        similarity (float): border similarity score
    """
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
    elif dir == DIR['l']:
        return 1/(1+np.mean(np.linalg.norm(np.subtract(img2[:,-1], img1[:,0]), axis = 1)))
    else:
        return -1
