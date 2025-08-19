# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:20:30 2023

#To do:
#self.GetMask -> Add reading image from url form web


@author: dvorr
"""
import cv2
import numpy as np
import os
from PIL import Image as PImage
import pkg_resources
import segmentation_models_pytorch as smp
import torch
from torchvision import transforms as T

from crackest.crack_pattern_analysis import CrackAnalyzer
from crackest.crack_plot import CrackPlot
from crackest.model_downloader import download_model, ONLINE_CRACKPY_MODELS

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CrackPy(CrackPlot):
    def __init__(self, model=1, model_path=None, model_type=None):
        self.impath = ""
        self.cran = CrackAnalyzer(self)
        # self.plot_app = CrackPlot(self)
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

        if model_type is None:
            self.model_type = "resnext50_32x4d"
        else:
            self.model_type = model_type

        if model_path is None:
            download_model(str(model))
            self.default_model = pkg_resources.resource_filename(
                "crackpy_models",
                r"{:s}".format(ONLINE_CRACKPY_MODELS[str(model)]),
            )
            self.model_path = "{}".format(self.default_model)
        else:
            self.model_path = model_path

        print(self.model_type)

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


    # def preview(self, mask=None):
    #     if self.has_mask == True:
    #         if mask is not None:
    #             self.plot_app.show_mask(mask)
    #             return

    #         self.plot_app.overlay()
    #     else:
    #         print("First extract mask")

    def get_img(self, impath):
        self.impath = impath
        self.hasimpath = True
        self.__read_img__()

    def set_cropdim(self, dim):
        self.crop_rec = dim
        self.crop = True

    def crop_img(self):
        if self.crop == True:
            dim = self.crop_rec
            imgo = self.img[dim[0] : dim[1], dim[2] : dim[3]]
            self.img_crop = imgo
            if self.has_mask == True:
                self.mask = self.mask[dim[0] : dim[1], dim[2] : dim[3]]

    def iterate_mask(self):
        if self.crop == False:
            imgo = self.img
        else:
            imgo = self.img_crop

        if self.gamma_correction is not None:
            imgo = self.__adjust_gamma__(imgo)

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
        self.masks = self.separate_mask(self.mask)

    def classify_img(self, impath):
        self.impath = impath
        img = cv2.imread(self.impath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_NEAREST)
        img = PImage.fromarray(img)
        self.img = img
        self.mask = self.__predict_image__(self.img)
        return self.mask

    def get_mask(self, impath=None, img=None, gamma=None, black_level=None):
        self.mm_ratio_set = False
        if impath is not None:
            self.impath = impath
            self.__read_img__()
        elif (impath is None) & (img is not None):
            self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.imgo = self.img
            self.crop = False
            self.img_read = True
        elif self.img_read == True:  # Img already read?
            pass

        self.gamma_correction = gamma
        self.black_level = black_level

        self.iterate_mask()

    def set_ratio(self, length=None, width=None):
        self.cran.set_ratio(length=None, width=None)

    def sep_masks(self):
        self.masks = self.separate_mask(self.mask)
        return self.masks

    def list_labels(self):
        labels = ["back", "spec", "mat", "crack", "pore"]
        return labels

    def get_metrics(self):
        self.sep_masks()
        self.cran.node_analysis()
        self.cran.basic_cnn_metrics()
        return self.cran.metrics.copy()

    def __loadmodel__(self):
        if self.is_cuda == True:
            self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
        else:
            self.model.load_state_dict(
                torch.load(
                    self.model_path,
                    map_location=self.device_type,
                    weights_only=True,
                )
            )
        self.model.eval()

    def __read_img__(self):

        img = cv2.imread(self.impath, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = img

        self.crop = False
        self.img_read = True
        self.has_mask = False

        self.mask = []

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

    def __predict_image__(self, image):
        self.model.eval()
        t = T.Compose([T.ToTensor(), T.Normalize(self.pred_mean, self.pred_std)])
        image = t(image)
        self.model.to(self.device)
        image = image.to(self.device)
        with torch.no_grad():
            image = image.unsqueeze(0)
            output = self.model(image)

            masked = torch.argmax(output, dim=1)
            masked = masked.cpu().squeeze(0)
        return masked

    def separate_mask(self, mask):
        back_bw = mask[:, :] == 0
        spec_bw = ~back_bw

        spec_bw = spec_bw.astype(np.uint8)
        back_bw = back_bw.astype(np.uint8)

        mat_bwo = mask[:, :] == 1
        mat_bwo = mat_bwo.astype(np.uint8)

        crack_bw = mask[:, :] == 2
        crack_bw = crack_bw.astype(np.uint8)

        pore_bw = mask[:, :] == 3
        pore_bw = pore_bw.astype(np.uint8)
        masks = {
            "back": back_bw,
            "spec": spec_bw,
            "mat": mat_bwo,
            "crack": crack_bw,
            "pore": pore_bw,
        }
        return masks
