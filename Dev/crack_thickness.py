# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:08:04 2024

@author: Richard
"""

import cv2
import numpy as np
from skimage.morphology import medial_axis
from skimage import img_as_ubyte

delta = 3  # delta index for interpolation

# get crack
im = cv2.imread("Img\TMpxX.jpg", cv2.IMREAD_GRAYSCALE)
rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)  # rgb just for demo purpose
_, crack = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)

# get medial axis
medial, distance = medial_axis(im, return_distance=True)
med_img = img_as_ubyte(medial)
med_contours, _ = cv2.findContours(med_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(rgb, med_contours, -1, (255, 0, 0), 1)
med_pts = [v[0] for v in med_contours[0]]

# get point with maximal distance from medial axis
max_idx = np.argmax(distance)
max_pos = np.unravel_index(max_idx, distance.shape)
max_dist = distance[max_pos]
coords = np.array([max_pos[1], max_pos[0]])
print(f"max distance from medial axis to boundary = {max_dist} at {coords}")

# interpolate orthogonal of medial axis at coords
idx = next(i for i, v in enumerate(med_pts) if (v == coords).all())
px1, py1 = med_pts[(idx-delta) % len(med_pts)]
px2, py2 = med_pts[(idx+delta) % len(med_pts)]
orth = np.array([py1 - py2, px2 - px1]) * max(im.shape)

# intersect orthogonal with crack and get contour
orth_img = np.zeros(crack.shape, dtype=np.uint8)
cv2.line(orth_img, coords + orth, coords - orth, color=255, thickness=1)
gap_img = cv2.bitwise_and(orth_img, crack)
gap_contours, _ = cv2.findContours(gap_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
gap_pts = [v[0] for v in gap_contours[0]]

# determine the end points of the gap contour by negative dot product
n = len(gap_pts)
gap_ends = [
    p for i, p in enumerate(gap_pts)
    if np.dot(p - gap_pts[(i-1) % n], gap_pts[(i+1) % n] - p) < 0
]
print(f"Maximum gap found from {gap_ends[0]} to {gap_ends[1]}")
cv2.line(rgb, gap_ends[0], gap_ends[1], color=(0, 0, 255), thickness=1)

cv2.imwrite("test_out.png", rgb)

#%%