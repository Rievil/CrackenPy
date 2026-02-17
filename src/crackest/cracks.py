# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from PIL import Image as PImage
import pkg_resources
import segmentation_models_pytorch as smp
import torch

from crackest.crack_pattern_analysis import CrackAnalyzer
from crackest.crack_plot import CrackPlot
from crackest.model_downloader import download_model, ONLINE_CRACKPY_MODELS

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CrackPy(CrackPlot):
    def __init__(
        self,
        model=1,
        model_dict=None,
        model_path=None,
        model_type=None,
        class_num=5,
        encoder_depth=5,
        img_channels=3,
    ):
        self.impath = ""
        self.cran = CrackAnalyzer(self)
        self.is_cuda = torch.cuda.is_available()

        if torch.backends.mps.is_available():
            self.device_type = "mps"
        elif torch.cuda.is_available():
            self.device_type = "cuda"
        else:
            self.device_type = "cpu"

        self.device = torch.device(self.device_type)

        if img_channels is not None:
            self.img_channels = img_channels
        else:
            self.img_channels = 3

        if encoder_depth is not None:
            self.encoder_depth = encoder_depth
        else:
            self.encoder_depth = 5

        if class_num is not None:
            self.class_num = class_num
        else:
            self.class_num = 5

        if model_type is None:
            self.model_type = "resnext50_32x4d"
        else:
            self.model_type = model_type
        print(self.model_type)
        self.reg_props = (
            "area",
            "centroid",
            "orientation",
            "axis_major_length",
            "axis_minor_length",
        )

        self.pred_mean = [0.485, 0.456, 0.406]
        self.pred_std = [0.229, 0.224, 0.225]
        self.batch_size = 16
        self.use_amp = self.device_type == "cuda"
        self.norm_mean = torch.tensor(self.pred_mean, dtype=torch.float32).view(
            1, len(self.pred_mean), 1, 1
        )
        self.norm_std = torch.tensor(self.pred_std, dtype=torch.float32).view(
            1, len(self.pred_std), 1, 1
        )
        self.norm_mean = self.norm_mean.to(self.device)
        self.norm_std = self.norm_std.to(self.device)

        if self.is_cuda == True:
            torch.backends.cudnn.benchmark = True

        self.patch_size = 416
        self.crop = False
        self.img_read = False
        self.hasimpath = False
        self.pixel_mm_ratio = 1
        self.mm_ratio_set = False
        self.has_mask = False
        self.gamma_correction = 1
        self.black_level = 1

        if model_dict is not None:
            # ===== Loading from custom model dict =====
            self.model_dict = model_dict
            self.state_dict = model_dict["state_dict"]
            self.config = model_dict["config"]
            self.model = smp.FPN(
                self.config["encoder_name"],
                in_channels=self.config["in_channels"],
                classes=self.config["num_classes"],
                activation=None,
                encoder_depth=self.encoder_depth,
            )
            self.model.load_state_dict(self.state_dict)
            self.model.to(self.device)
            self.model.eval()
            return

        else:
            # ===== Loading from web by default =====
            if model_path is None:
                download_model(str(model))
                self.default_model = pkg_resources.resource_filename(
                    "crackpy_models",
                    r"{:s}".format(ONLINE_CRACKPY_MODELS[str(model)]),
                )
                self.model_path = "{}".format(self.default_model)
            else:
                self.model_path = model_path

            self.model = smp.FPN(
                self.model_type,
                in_channels=self.img_channels,
                classes=self.class_num,
                activation=None,
                encoder_depth=self.encoder_depth,
            )
            self.__loadmodel__()

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

    def iterate_mask(self, batch_size=None):
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
        if batch_size is None:
            batch_size = self.batch_size
        else:
            batch_size = max(1, int(batch_size))
            self.batch_size = batch_size

        blank_image = np.zeros((int(sz[0]), int(sz[1])), np.uint8)

        xstarts = self.__tile_starts__(sz[0], step_size)
        ystarts = self.__tile_starts__(sz[1], step_size)

        batch_tiles = []
        batch_coords = []

        for xstart in xstarts:
            xstop = min(xstart + step_size, sz[0])
            for ystart in ystarts:
                ystop = min(ystart + step_size, sz[1])
                batch_tiles.append(imgo[xstart:xstop, ystart:ystop])
                batch_coords.append((xstart, xstop, ystart, ystop))

                if len(batch_tiles) == batch_size:
                    batch_masks = self.__predict_batch__(batch_tiles)
                    for mask, coord in zip(batch_masks, batch_coords):
                        xs, xe, ys, ye = coord
                        blank_image[xs:xe, ys:ye] = mask
                    batch_tiles = []
                    batch_coords = []

        if len(batch_tiles) > 0:
            batch_masks = self.__predict_batch__(batch_tiles)
            for mask, coord in zip(batch_masks, batch_coords):
                xs, xe, ys, ye = coord
                blank_image[xs:xe, ys:ye] = mask

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

    def get_mask(
        self, impath=None, img=None, gamma=None, black_level=None, batch_size=None
    ):
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

        self.iterate_mask(batch_size=batch_size)

    def set_ratio(self, length=None, width=None):
        self.cran.set_ratio(length=length, width=width)

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
        if self.mm_ratio_set == True:
            self.cran.metrics["ratio_mm2px"]
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
        self.model.to(self.device)
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
        if self.is_cuda == True:
            torch.cuda.empty_cache()

    def __predict_image__(self, image):
        mask = self.__predict_batch__([image])[0]
        return torch.from_numpy(mask.astype(np.int64))

    def __tile_starts__(self, dim_size, step_size):
        if dim_size <= step_size:
            return [0]

        starts = list(range(0, dim_size - step_size + 1, step_size))
        last_start = dim_size - step_size
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def __prepare_image__(self, image):
        if isinstance(image, PImage.Image):
            image = np.asarray(image)
        elif torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        else:
            image = np.asarray(image)

        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        return image

    def __predict_batch__(self, images):
        self.model.eval()

        if isinstance(images, np.ndarray) and images.ndim == 4:
            image_batch = images
        else:
            prepared = [self.__prepare_image__(image) for image in images]
            image_batch = np.stack(prepared, axis=0)

        image_batch = np.ascontiguousarray(image_batch)
        image_batch = torch.from_numpy(image_batch).permute(0, 3, 1, 2).float()
        image_batch = image_batch.div(255.0)
        image_batch = image_batch.to(self.device, non_blocking=self.is_cuda)

        if image_batch.shape[1] == self.norm_mean.shape[1]:
            image_batch = (image_batch - self.norm_mean) / self.norm_std

        with torch.inference_mode():
            if self.use_amp == True:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(image_batch)
            else:
                output = self.model(image_batch)

            masked = torch.argmax(output, dim=1)
            masked = masked.to(dtype=torch.uint8).cpu().numpy()

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
