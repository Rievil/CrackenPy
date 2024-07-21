# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:20:30 2023

#To do:
#self.GetMask -> Add reading image from url form web


@author: dvorr
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import torch

from torchvision import transforms as T


from PIL import Image as PImage
import cv2

import time
import os
from tqdm.notebook import tqdm


import segmentation_models_pytorch as smp

# from wand.image import Image as WI
from skimage.morphology import medial_axis, skeletonize

from skimage.measure import label, regionprops, regionprops_table
import math
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import pdist
import pkg_resources

import gdown
import crackpy_models

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ONLINE_CRACKPY_MODELS = {
    "0": [
        "resnext101_32x8d_N387_C5_30102023.pt",
        "1AtTrLmDf7kmlfEbGEJ5e43_aa0SnGntL",
    ],
    "1": [
        "resnext101_32x8d_N387_C5_310124.pt",
        "1qmAv34aIPRLCRGEG3gwbbsYQTYmZpnp5",
    ],
}


def DownloadModel(key):
    model = pkg_resources.resource_listdir("crackpy_models", "")
    online_models = ONLINE_CRACKPY_MODELS
    count = model.count(online_models[key][0])

    if count == 0:
        module_path = crackpy_models.__file__
        tar_folder = os.path.dirname(module_path)

        if tar_folder.count("/") > 0:
            out_file = r"{:s}/{:s}".format(tar_folder, online_models[key][0])
        else:
            out_file = r"{:s}\{:s}".format(tar_folder, online_models[key][0])

        url_id = online_models[key][1]
        print(
            "Downloading deep learing model '{:s}' for module crackpy".format(
                online_models[key][0].replace(".pt", "")
            )
        )
        gdown.download(id=url_id, output=out_file, quiet=False)


def UpdateModels():
    model = pkg_resources.resource_listdir("crackpy_models", "")
    online_models = ONLINE_CRACKPY_MODELS

    count_d = 0
    for key in online_models:
        count = model.count(online_models[key][0])
        if count == 0:
            count_d += 1
            DownloadModel(key)

    if count_d == 0:
        print("All models are already downloaded")
    else:
        print("Downloaded {:d} models".format(count_d))
    pass


class CrackPy:
    def __init__(self, model=1):
        self.impath = ""

        self.is_cuda = torch.cuda.is_available()
        if torch.backends.mps.is_available():
            self.device_type = "mps"
        elif torch.cuda.is_available():
            self.device_type = "cuda"
        else:
            self.device_type = "cpu"

        self.device = torch.device(self.device_type)

        self.img_channels = 3
        self.encoder_depth = 5
        self.class_num = 5

        self.model_type = "resnext101_32x8d"

        DownloadModel(str(model))
        self.default_model = pkg_resources.resource_filename(
            "crackpy_models",
            r"{:s}".format(ONLINE_CRACKPY_MODELS[str(model)][0]),
        )

        self.model_path = "{}".format(self.default_model)

        self.model = smp.FPN(
            self.model_type,
            in_channels=self.img_channels,
            classes=self.class_num,
            activation=None,
            encoder_depth=self.encoder_depth,
        )

        # self.model_path=
        self.__loadmodel__()
        self.reg_props = (
            "area",
            "centroid",
            "orientation",
            "axis_major_length",
            "axis_minor_length",
        )
        self.pred_mean = [0.485, 0.456, 0.406]
        self.pred_std = [0.229, 0.224, 0.225]
        self.patch_size = 416
        self.crop = False
        self.img_read = False
        self.hasimpath = False
        self.pixel_mm_ratio = 1
        self.mm_ratio_set = False
        self.has_mask = False
        self.gamma_correction = 1
        self.black_level = 1

        pass

    def GetImg(self, impath):
        self.impath = impath
        self.hasimpath = True
        self.__ReadImg__()

    def __loadmodel__(self):
        if self.is_cuda == True:
            self.model.load_state_dict(
                torch.load(self.model_path, weights_only=True)
            )
        else:
            self.model.load_state_dict(
                torch.load(
                    self.model_path,
                    map_location=self.device_type,
                    weights_only=True,
                )
            )
        self.model.eval()

    def __ReadImg__(self):
        # if ".heic" in self.impath.lower():
        #     img = WI(filename=self.impath)
        #     img.format = "jpg"
        #     img_buffer = np.asarray(bytearray(img.make_blob()), dtype=np.uint8)
        #     img = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)
        # else:

        img = cv2.imread(self.impath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = img

        self.crop = False
        self.img_read = True
        self.has_mask = False

        self.mask = []
        # return img

    def ClassifyImg(self, impath):
        self.impath = impath
        img = cv2.imread(self.impath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_NEAREST)
        img = PImage.fromarray(img)
        self.img = img
        self.mask = self.__predict_image__(self.img)
        self.img
        return self.mask

    def GetMask(self, impath=None, img=None, gamma=None, black_level=None):
        self.mm_ratio_set = False
        if impath is not None:
            self.impath = impath
            self.__ReadImg__()
        elif (impath is None) & (img is not None):
            self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.imgo = self.img
            self.crop = False
            self.img_read = True
        elif self.img_read == True:  # Img already read?
            pass

        self.gamma_correction = gamma
        self.black_level = black_level

        self.IterateMask()

    def __black_level__(self, img):
        black_level = self.black_level
        image = img.astype("float32")

        # Apply black level correction
        corrected_image = image - black_level

        # Clip pixel values to ensure they stay within valid range [0, 255]
        corrected_image = np.clip(corrected_image, 0, 255)

        # Convert back to uint8
        corrected_image = corrected_image.astype("uint8")
        return corrected_image

    def __adjust_gamma__(self, img):
        gamma = self.gamma_correction
        invGamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")

        return cv2.LUT(img, table)

    def __del__(self):
        torch.cuda.empty_cache()

    def SetRatio(self, length=None, width=None):
        self.mm_ratio_set = True
        reg_props = (
            "area",
            "centroid",
            "orientation",
            "axis_major_length",
            "axis_minor_length",
        )

        if length is None:
            self.length = 160
        else:
            self.length = length

        if width is None:
            self.width = 40
        else:
            self.width = width

        mask = np.array(self.mask)
        bw_mask = mask[:, :] == 1
        image = bw_mask.astype(np.uint8)
        label_img = label(image)

        props_mat = regionprops_table(label_img, properties=reg_props)

        self.orientation = props_mat["orientation"]

        dfmat = pd.DataFrame(props_mat)
        dfmat.sort_values(by=["area"], ascending=False)
        dfmat = dfmat.reset_index()

        l_rat = self.length / dfmat["axis_major_length"][0]
        w_rat = self.width / dfmat["axis_minor_length"][0]
        m_rat = (l_rat + w_rat) / 2
        self.pixel_mm_ratio = m_rat
        return self.pixel_mm_ratio

    def __predict_image__(self, image):
        self.model.eval()
        t = T.Compose(
            [T.ToTensor(), T.Normalize(self.pred_mean, self.pred_std)]
        )
        image = t(image)
        self.model.to(self.device)
        image = image.to(self.device)
        with torch.no_grad():
            image = image.unsqueeze(0)
            output = self.model(image)

            masked = torch.argmax(output, dim=1)
            masked = masked.cpu().squeeze(0)
        return masked

    def SepMasks(self):
        self.__SeparateMask__()
        return self.masks

    def __SeparateMask__(self):
        back_bw = self.mask[:, :] == 0
        spec_bw = ~back_bw

        spec_bw = spec_bw.astype(np.uint8)
        back_bw = back_bw.astype(np.uint8)

        mat_bwo = self.mask[:, :] == 1
        mat_bwo = mat_bwo.astype(np.uint8)

        crack_bw = self.mask[:, :] == 2
        crack_bw = crack_bw.astype(np.uint8)

        pore_bw = self.mask[:, :] == 3
        pore_bw = pore_bw.astype(np.uint8)

        self.masks = {
            "back": back_bw,
            "spec": spec_bw,
            "mat": mat_bwo,
            "crack": crack_bw,
            "pore": pore_bw,
        }

    def MeasureBW(self):
        self.__SeparateMask__()

        kernel = np.ones((50, 50), np.uint8)
        mat_bw = cv2.dilate(self.masks["mat"], kernel, iterations=1)
        mat_bw = cv2.erode(mat_bw, kernel)

        crack_bw = cv2.bitwise_and(mat_bw, self.masks["crack"])
        pore_bw = cv2.bitwise_and(mat_bw, self.masks["pore"])

        total_area = self.masks["back"].shape[0] * self.masks["back"].shape[1]
        back_area = self.masks["back"].sum()
        spec_area = total_area - back_area
        crack_area = crack_bw.sum()
        pore_area = pore_bw.sum()

        mat_area = total_area - (crack_area + spec_area + pore_area)

        crack_ratio = crack_area / spec_area

        skel = skeletonize(crack_bw, method="lee")

        crack_length = skel.sum()
        crack_avg_thi = crack_area / crack_length

        result = {
            "spec_area": spec_area * self.pixel_mm_ratio,
            "mat_area": mat_area * self.pixel_mm_ratio,
            "crack_area": crack_area * self.pixel_mm_ratio,
            "crack_ratio": crack_ratio,
            "crack_length": crack_length * self.pixel_mm_ratio,
            "crack_thickness": crack_avg_thi * self.pixel_mm_ratio,
            "pore_area": pore_area * self.pixel_mm_ratio,
        }

        self.bw_stats = result
        self.__MeasurePores__()
        return result

    def __MeasurePores__(self):
        image_pore = self.masks["pore"]
        label_img_pore = label(image_pore)

        props_pore = regionprops_table(
            label_img_pore, properties=self.reg_props
        )
        dfpores = pd.DataFrame(props_pore)

        mask = dfpores["area"] < 10
        dfpores = dfpores[~mask]

        dfpores.sort_values(by=["area"], ascending=False)
        dfpores = dfpores.reset_index()

        points = np.array([dfpores["centroid-1"], dfpores["centroid-0"]])
        points = np.rot90(points)
        arr = pdist(points, metric="minkowski")

        avgdist = arr.mean()
        area = dfpores["area"].mean()
        self.bw_stats["avg_pore_distance"] = avgdist
        self.bw_stats["avg_pore_size"] = area

    def SetCropDim(self, dim):
        self.crop_rec = dim
        self.crop = True

    def CropImg(self):
        if self.crop == True:
            dim = self.crop_rec
            imgo = self.img[dim[0] : dim[1], dim[2] : dim[3]]
            self.img_crop = imgo
            if self.has_mask == True:
                self.mask = self.mask[dim[0] : dim[1], dim[2] : dim[3]]

    def IterateMask(self):
        if self.crop == False:
            imgo = self.img
        else:
            imgo = self.img_crop

        if self.gamma_correction is not None:
            imgo = self.__adjust_gamma__(imgo)
            # self.img=self.__adjust_gamma__()

        if self.black_level is not None:
            imgo = self.__black_level__(imgo)

        sz = imgo.shape
        step_size = self.patch_size

        xcount = sz[0] / step_size
        xcount_r = np.ceil(xcount)
        ycount = sz[1] / step_size
        ycount_r = np.ceil(ycount)

        blank_image = np.zeros((int(sz[0]), int(sz[1])), np.uint8)

        width = step_size
        height = width

        for xi in range(0, int(xcount_r)):
            for yi in range(0, int(ycount_r)):
                if xi < xcount - 1:
                    xstart = width * xi
                    xstop = xstart + width
                else:
                    xstop = sz[0]
                    xstart = xstop - step_size

                if yi < ycount - 1:
                    ystart = height * yi
                    ystop = ystart + height
                else:
                    ystop = sz[1]
                    ystart = ystop - step_size

                cropped_image = imgo[xstart:xstop, ystart:ystop]

                mask = self.__predict_image__(cropped_image)
                blank_image[xstart:xstop, ystart:ystop] = mask

        self.mask = blank_image
        self.has_mask = True
        self.__SeparateMask__()


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle


class CrackPlot:
    def __init__(self, crackpy):
        self.CP = crackpy
        colors = ["#25019E", "#717171", "#CD0000", "#ECFF00"]
        my_cmap = ListedColormap(colors, name="my_cmap")
        self.cmap = my_cmap

    def show_img(self):
        fig, ax = plt.subplots(1, 1)

        if self.CP.crop == False:
            ax.imshow(self.CP.img)
        else:
            ax.imshow(self.CP.img_crop)

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        self.fig = fig

    def overlay(self, figsize=[5, 4]):
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax = plt.gca()
        if self.CP.crop == False:
            ax.imshow(self.CP.img)
        else:
            ax.imshow(self.CP.img_crop)

        im = ax.imshow(self.CP.mask, alpha=0.8, cmap="jet")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        cbar = plt.colorbar(im, cax=cax)
        cbar.set_ticks([0, 1, 2, 3])
        cbar.ax.set_yticklabels(["Back", "Matrix", "Crack", "Pore"])
        cbar.ax.tick_params(labelsize=10, size=0)

        ax.axis("off")
        plt.show()
        self.fig = fig

    def Save(self, name):
        self.fig.savefig(
            "{:s}".format(name), dpi=300, bbox_inches="tight", pad_inches=0
        )

    def distancemap(self):
        mask = self.CP.mask
        crack_bw = mask[:, :] == 2
        crack_bw = crack_bw.astype(np.uint8)

        thresh = crack_bw
        # Determine the distance transform.
        skel = skeletonize(thresh, method="lee")
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        idx = skel == 1
        dist_skel = dist[idx]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))

        ax.imshow(self.CP.img)

        if self.CP.mm_ratio_set == True:
            im = ax.imshow(
                dist * self.CP.pixel_mm_ratio, cmap="jet", alpha=0.8
            )
        else:
            im = ax.imshow(dist, cmap="jet", alpha=0.8)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=10, size=0)

        ax.axis("off")

        if self.CP.mm_ratio_set == True:
            arr_dist = dist[skel == 1] * 2 * self.CP.pixel_mm_ratio
            plt.suptitle("Mean thickness {:.2f} mm".format(arr_dist.mean()))
            cbar.ax.set_label("Thickness [mm]")
        else:
            arr_dist = dist[skel == 1] * 2
            plt.suptitle(
                "Mean thickness {:.2f} pixels".format(arr_dist.mean())
            )
            cbar.ax.set_ylabel("Thickness [px]")

        plt.show()
        self.fig = fig
