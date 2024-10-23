"""Based on the Huggingface Course for Computer Vision. See, https://huggingface.co/learn/computer-vision-course/unit1/feature-extraction/feature-matching"""

#
# For this script you need to add two images into the folder "HF_CV_Course/imgs". The second one should be a element of the first image.
#

import cv2
import time
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia_moons.viz import draw_LAF_matches


def time_function(output_label, f, *args, **kwargs):
    """Time the excecution of a function. The function output is returned.
    This functions is not very accurate because the function is only excecuted once.
    """
    first_time = time.time_ns()
    result = f(*args, **kwargs)
    second_time = time.time_ns()
    print(f"{output_label}: {int((second_time - first_time)/1000):_d} us")
    return result


img1 = cv2.cvtColor(cv2.imread("HF_CV_Course/imgs/img1.png"), cv2.COLOR_RGB2GRAY)
img2 = cv2.cvtColor(cv2.imread("HF_CV_Course/imgs/img2.png"), cv2.COLOR_RGB2GRAY)

# To see images uncomment this, the third line prevents the process to exit and close the images
# cv2.imshow("Image 1", img1)
# cv2.imshow("Image 2", img2)
# cv2.waitKey(0)

sift = cv2.SIFT.create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Brute force feature matching
bf = cv2.BFMatcher()
matches = time_function("Bruteforce Duration", bf.knnMatch, des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.5 * n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(
    img1,
    kp1,
    img2,
    kp2,
    good,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

# cv2.imshow("SIFT Brute Force Result", img3)


# Brute Force with ORB (binary) discriptor
orb = cv2.ORB.create(1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = time_function("Orb Brute Force", bf.match, des1, des2)  # bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

n = 1000

img3 = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    matches[:n],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

# cv2.imshow("ORB Brute Force Result", img3)


# FLANN

# Dictionary specifing algorithm
# SIFT, SURF
FLANN_INDEX_KDTREE = 1
sift_index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

# ORB
FLANN_INDEX_LSH = 6
orb_index_params = dict(
    algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=2
)

# Max number of leafs to visit
search_params = dict(checks=50)

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

flann = cv2.FlannBasedMatcher(sift_index_params, search_params)
matches = time_function(
    "SIFT Features FLANN", flann.knnMatch, des1, des2, k=2
)  # flann.knnMatch(des1, des2, k=2)


# Draw only good matches
matchesMask = [[0, 0] for i in range(len(matches))]
for i, (m, n) in enumerate(matches):
    if m.distance < 0.5 * n.distance:
        matchesMask[i] = [1, 0]

draw_params = dict(
    matchColor=(0, 255, 0),
    singlePointColor=(255, 0, 0),
    matchesMask=matchesMask,
    flags=cv2.DrawMatchesFlags_DEFAULT,
)

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
# cv2.imshow("FLANN with SIFT Features", img3)


img1 = K.io.load_image("./HF_CV_Course/imgs/img1.png", K.io.ImageLoadType.RGB32)[
    None, ...
]
img2 = K.io.load_image("./HF_CV_Course/imgs/img2.png", K.io.ImageLoadType.RGB32)[
    None, ...
]

img1 = K.geometry.resize(img1, (512, 512), antialias=True)
img2 = K.geometry.resize(img2, (512, 512), antialias=True)

matcher = KF.LoFTR(pretrained="outdoor")

input_dict = {
    "image0": K.color.rgb_to_grayscale(img1),
    "image1": K.color.rgb_to_grayscale(img2),
}

with torch.inference_mode():
    correspondences = matcher(input_dict)

mkpts0 = correspondences["keypoints0"].cpu().numpy()
mkpts1 = correspondences["keypoints1"].cpu().numpy()
Fm, inliers = cv2.findFundamentalMat(
    mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000
)
inliers = inliers > 0

print("Draw LoFTR")
draw_LAF_matches(
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts0).view(1, -1, 2),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1),
    ),
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts1).view(1, -1, 2),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1),
    ),
    torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
    K.tensor_to_image(img1),
    K.tensor_to_image(img2),
    inliers,
    draw_dict={
        "inlier_color": (0.1, 1, 0.1, 0.5),
        "tentative_color": None,
        "feature_color": (0.2, 0.2, 1, 0.5),
        "vertical": False,
    },
)

# cv2.waitKey(0)
