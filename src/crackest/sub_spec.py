import cv2
import numpy as np
from skimage import measure

from crackest.crack_pattern_analysis import CrackAnalyzer
from crackest.crack_plot import CrackPlot


class SubSpec(CrackPlot):
    def __init__(self, img, mask: dict, full_mask, sett):
        self.img = img
        self.mask = full_mask
        self.masks = mask  # dictionary
        self.sett = sett
        self.cran = CrackAnalyzer(self)

    def get_metrics(self):
        self.cran.node_analysis()
        self.cran.basic_cnn_metrics()

        return self.cran.metrics

    def set_ratio(self, length: float = 160, width: float = 40, ratio: int = 1):
        self.length = length
        self.width = width
        self.cran.set_ratio(self.length, self.width)

    def get_countours(self):
        r = self.masks["spec"]

        total_area = r.shape[0] * r.shape[1]
        area_trsh = int(total_area * 0.3)

        kernel = np.ones((20, 20), np.uint8)
        r = cv2.erode(r, kernel)
        r = cv2.dilate(r, kernel, iterations=1)

        contours = measure.find_contours(r, 0.8)

        area = 0
        for i in range(len(contours)):
            count = contours[i]
            c = np.expand_dims(count.astype(np.float32), 1)
            c = cv2.UMat(c)
            area = cv2.contourArea(c)
            if area > area_trsh:
                break

        image_height, image_width = (
            r.shape[0],
            r.shape[1],
        )  # Replace with your actual image size
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        # Fill the area inside the contour with 1
        cv2.fillPoly(mask, [count], color=1)

        self.specimen_mask = mask
        self.area_treashold = area_trsh
        self.area = area
        self.contour = count
