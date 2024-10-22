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
from skimage import measure
import torch
import sknw
from torchvision import transforms as T
from PIL import Image as PImage
import cv2
import time
import os
from tqdm.notebook import tqdm


# from wand.image import Image as WI
from skimage.morphology import medial_axis, skeletonize

from skimage.measure import label, regionprops, regionprops_table
import math
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import pdist
import pkg_resources

import gdown
import crackpy_models


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
import segmentation_models_pytorch as smp

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ONLINE_CRACKPY_MODELS = {
    "0": "model1.pt",
    "1": "model2.pt",
}


def DownloadModel(key):
    from huggingface_hub import hf_hub_download

    model = pkg_resources.resource_listdir("crackpy_models", "")
    online_models = ONLINE_CRACKPY_MODELS
    count = model.count(online_models[key])

    if count == 0:
        module_path = crackpy_models.__file__
        tar_folder = os.path.dirname(module_path)

        print(
            "Downloading deep learing model '{:s}' for module crackpy".format(
                online_models[key].replace(".pt", "")
            )
        )
        hf_hub_download(
            repo_id="rievil/crackenpy",
            filename=online_models[key],
            local_dir=tar_folder,
        )
        print("... done downloading")
        # gdown.download(id=url_id, output=out_file, quiet=False)


def UpdateModels():
    model = pkg_resources.resource_listdir("crackpy_models", "")
    online_models = ONLINE_CRACKPY_MODELS

    count_d = 0
    for key in online_models:
        count = model.count(online_models[key])
        if count == 0:
            count_d += 1
            DownloadModel(key)

    if count_d == 0:
        print("All models are already downloaded")
    else:
        print("Downloaded {:d} models".format(count_d))
    pass


class CrackPlot:
    def __init__(self):
        self.colors = ["#25019E", "#717171", "#CD0000", "#ECFF00"]
        self.class_names = ["back", "matrix", "crack", "pore"]
        self.cmap = ListedColormap(self.colors, name="my_cmap")

    def show_img(self):
        fig, ax = plt.subplots(1, 1)

        ax.imshow(self.img)

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        self.ax = ax
        self.fig = fig

    def show_mask(self, mask="crack"):
        fig, ax = plt.subplots(1, 1)

        ax.imshow(self.masks[mask], alpha=0.8)

        ax.set_title("Showing mask: {:s}".format(mask))

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        plt.tight_layout()
        self.ax = ax
        self.fig = fig

    def overlay(self, figsize=[5, 4]):
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax = plt.gca()

        ax.imshow(self.img)

        im = ax.imshow(self.mask, alpha=0.8, cmap="jet")

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
        self.ax = ax
        self.fig = fig

    def save(self, name):
        self.fig.savefig(
            "{:s}".format(name), dpi=300, bbox_inches="tight", pad_inches=0
        )

    def distancemap(self):
        thresh = self.masks["crack"]
        # Determine the distance transform.
        skel = skeletonize(thresh, method="lee")
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        idx = skel == 1
        dist_skel = dist[idx]

        fig, ax = plt.subplots(nrows=1, ncols=1)

        ax.imshow(self.img)

        if self.cran.pixel_mm_ratio_set == True:
            im = ax.imshow(dist * self.cran.pixel_mm_ratio, cmap="jet", alpha=0.8)
        else:
            im = ax.imshow(dist, cmap="jet", alpha=0.8)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=10, size=0)

        ax.axis("off")

        if self.cran.pixel_mm_ratio_set == True:
            arr_dist = dist[skel == 1] * 2 * self.cran.pixel_mm_ratio
            ax.set_title("Mean thickness {:.2f} mm".format(arr_dist.mean()))
            cbar.ax.set_label("Thickness [mm]")
        else:
            arr_dist = dist[skel == 1] * 2
            ax.set_title("Mean thickness {:.2f} pixels".format(arr_dist.mean()))
            cbar.ax.set_ylabel("Thickness [px]")

        plt.tight_layout()
        plt.show()
        self.ax = ax
        self.fig = fig

    def __anotate_img__(self, img, prog, label):
        img2 = img.copy()
        font = cv2.FONT_HERSHEY_DUPLEX

        color = (255, 255, 255)

        new_image_width = 300
        new_image_height = 300
        color = (255, 0, 0)

        fontScale = 2
        thickness = 3
        frame = 50
        height = 40
        bar_font_space = 30
        bwspace = 8

        wi, he, channels = img2.shape

        color = (0, 0, 0)
        result = np.full(
            (wi + (frame + height + bar_font_space + 80), he, channels),
            color,
            dtype=np.uint8,
        )

        result[0:wi, 0:he, :] = img2
        img2 = result

        wi, he, channels = img2.shape

        startp = [frame, wi - frame]
        endp = [he - frame, wi - (frame + height)]

        text_point = (frame + 120, wi - (frame + height + bar_font_space))

        img2 = cv2.putText(
            img2,
            label,
            text_point,
            font,
            fontScale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        startp_prog = [frame + bwspace, wi - frame - bwspace]
        endp_prog = [
            int((he - frame - bwspace) * prog),
            wi - (frame + height) + bwspace,
        ]

        xpoints = np.linspace(startp[0], endp[0] - 10, 11)

        img2 = cv2.rectangle(img2, startp, endp, color=(255, 255, 255), thickness=-1)
        if prog >= 0.01:
            img2 = cv2.rectangle(
                img2, startp_prog, endp_prog, color=(0, 0, 0), thickness=-1
            )

        # Ratio line
        ratio = self.subspec["ratio"]

        r_startp = [
            int(he - (frame + ratio * 40)),
            int(wi - (frame + height + bwspace * 3 + height)),
        ]
        r_endp = [he - frame, wi - (frame + height + bwspace * 3)]

        img2 = cv2.rectangle(img2, r_startp, r_endp, color=(150, 50, 50), thickness=-1)

        img2 = cv2.putText(
            img2,
            "40 mm",
            [r_startp[0] - 300, r_startp[1] + height],
            font,
            fontScale,
            (150, 50, 50),
            thickness,
            cv2.LINE_AA,
        )

        return img2


class CrackAnalyzer:
    """Analyze crack patterns in the crack graph."""

    def __init__(self, parent):
        self.parent = parent
        self.reg_props = (
            "area",
            "centroid",
            "orientation",
            "axis_major_length",
            "axis_minor_length",
            "bbox",
        )
        self.metrics = dict()
        self.pixel_mm_ratio = 1
        self.pixel_mm_ratio_set = False
        self.min_number_of_crack_points = 20
        self.has_contour = False

    def node_analysis(self):
        self.build_graph()
        df_nodes, df_edges = self.analyze_cracks()

        mean_angle_weighted = (df_edges["angle"] * df_edges["length"]).sum() / df_edges[
            "length"
        ].sum()
        self.metrics["edge_per_node"] = df_nodes["num_edges"].mean()
        self.metrics["crack_tot_length"] = df_edges["length"].sum()
        self.metrics["average_angle"] = mean_angle_weighted

    def get_countours(self):
        r = self.parent.masks["back"]
        r = (~r).astype(np.uint8)

        total_area = r.shape[0] * r.shape[1]
        area_trsh = int(total_area * 0.5)

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
        self.has_contour = True

    def set_ratio(self, length=None, width=None):
        mask = self.parent.masks["spec"]

        w, h = mask.shape

        hor_line = mask[int(w / 2 - 10) : int(w / 2 + 10), :].mean(axis=0)
        hind = np.where(hor_line > 0)[0]
        length_px = np.diff([hind[0], hind[-1]])
        len_rat = length / length_px

        ver_line = mask[:, int(h / 2 - 10) : int(h / 2 + 10)].mean(axis=1)
        vind = np.where(ver_line > 0)[0]
        self.hor_coor = [vind[0], vind[-1]]
        self.ver_coor = [hind[0], hind[-1]]

        width_px = np.diff([vind[0], vind[-1]])
        wid_rat = width / width_px
        self.pixel_mm_ratio = np.mean([len_rat, wid_rat])
        self.pixel_mm_ratio_set = True
        print("Pixel to mm ratio: {:0.2f} mm/px".format(self.pixel_mm_ratio))

    def get_equations(self):
        """Get main equations for main and secondary axis of the specimen"""
        # mask = np.array(cp.mask)
        bw_mask = self.parent.masks["back"]
        bw_mask = ~bw_mask

        image = bw_mask.astype(np.uint8)
        label_img = label(image)
        # regions = regionprops(label_img)

        props_mat = regionprops_table(label_img, properties=self.reg_props)
        dfmat = pd.DataFrame(props_mat)
        dfmat.sort_values(by=["area"], ascending=True)
        dfmat = dfmat.reset_index()

        #
        dfii = pd.DataFrame()
        for index, props in dfmat.iterrows():
            # y0, x0 = props.centroid-0
            if props["area"] > 2000:
                y0 = props["centroid-0"]
                x0 = props["centroid-1"]

                orientation = props["orientation"]

                rat1 = 0.43
                x0i = x0 - math.cos(orientation) * rat1 * props["axis_minor_length"]
                y0i = y0 + math.sin(orientation) * rat1 * props["axis_minor_length"]

                x1 = x0 + math.cos(orientation) * rat1 * props["axis_minor_length"]
                y1 = y0 - math.sin(orientation) * rat1 * props["axis_minor_length"]

                rat2 = 0.43
                x2i = x0 + math.sin(orientation) * rat2 * props["axis_major_length"]
                y2i = y0 + math.cos(orientation) * rat2 * props["axis_major_length"]

                x2 = x0 - math.sin(orientation) * rat2 * props["axis_major_length"]
                y2 = y0 - math.cos(orientation) * rat2 * props["axis_major_length"]

                he = {"alpha": (y0i - y1) / (x0i - x1)}
                he["beta"] = y1 - he["alpha"] * x1
                he["len"] = props["axis_minor_length"]
                he["label"] = "width"

                ve = {"alpha": (y2i - y2) / (x2i - x2)}
                ve["beta"] = y2 - ve["alpha"] * x2
                ve["len"] = props["axis_major_length"]
                ve["label"] = "length"

                minr = int(props["bbox-0"])
                minc = int(props["bbox-1"])
                maxr = int(props["bbox-2"])
                maxc = int(props["bbox-3"])

                bx = (minc, maxc, maxc, minc, minc)
                by = (minr, minr, maxr, maxr, minr)

        eq = {"h": he, "v": ve}
        xin = (eq["h"]["beta"] - eq["v"]["beta"]) / (
            eq["v"]["alpha"] - eq["h"]["alpha"]
        )
        yin = xin * eq["h"]["alpha"] + eq["h"]["beta"]
        eq["center"] = (xin, yin)
        return eq

    def build_graph(self):
        self.eq = self.get_equations()

        # Filter only cracks mask
        crack_bw = self.parent.masks["crack"]
        crack_bw = crack_bw.astype(np.uint8)

        # Determine the distance transform.
        self.crack_skeleton = skeletonize(crack_bw, method="lee")
        self.graph = sknw.build_sknw(self.crack_skeleton, multi=False)

    def __meas_pores__(self):
        image_pore = self.parent.masks["pore"]
        label_img_pore = label(image_pore)

        props_pore = regionprops_table(label_img_pore, properties=self.reg_props)
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

        self.metrics["avg_pore_distance"] = avgdist
        self.metrics["avg_pore_size"] = area

    def basic_cnn_metrics(self):
        kernel = np.ones((50, 50), np.uint8)
        mat_bw = cv2.dilate(self.parent.masks["mat"], kernel, iterations=1)
        mat_bw = cv2.erode(mat_bw, kernel)

        crack_bw = cv2.bitwise_and(mat_bw, self.parent.masks["crack"])
        pore_bw = cv2.bitwise_and(mat_bw, self.parent.masks["pore"])

        total_area = (
            self.parent.masks["back"].shape[0] * self.parent.masks["back"].shape[1]
        )
        back_area = self.parent.masks["back"].sum()
        spec_area = total_area - back_area
        crack_area = crack_bw.sum()
        pore_area = pore_bw.sum()

        mat_area = total_area - (crack_area + spec_area + pore_area)

        crack_ratio = crack_area / spec_area

        crack_length = self.crack_skeleton.sum()
        crack_avg_thi = crack_area / crack_length

        self.metrics["spec_area"] = (spec_area * self.pixel_mm_ratio,)
        self.metrics["mat_area"] = (mat_area * self.pixel_mm_ratio,)
        self.metrics["crack_ratio"] = (crack_ratio,)
        self.metrics["crack_length"] = (crack_length * self.pixel_mm_ratio,)
        self.metrics["crack_thickness"] = (crack_avg_thi * self.pixel_mm_ratio,)
        self.metrics["pore_area"] = (pore_area * self.pixel_mm_ratio,)

        self.__meas_pores__()

    def _analyze_edge(self, pts):
        length = 0
        angle_deg_length = 0
        for i in range(len(pts) - 1):
            seg_length, seg_angle_deg = self._analyze_crack_segment(pts[i], pts[i + 1])
            length += seg_length
            angle_deg_length += seg_angle_deg * seg_length
        angle_deg = angle_deg_length / length  # weighted mean

        return {
            "num_pts": len(pts),
            "length": length,
            "angle": angle_deg,
        }

    def _analyze_crack_segment(self, pt1, pt2):
        length = np.sqrt(np.sum(np.square(pt1 - pt2)))

        # crack angle: only positive angles are considered:
        #        90°
        # 180° __|__ 0°
        # V crack mean angle: (45+135) / 2 = 90°
        # A crack mean angle: ((180-135)+(180-45)) / 2  = (45+135) / 2 = 90°  (not -90°)
        # swapped X and Y coordinates?
        delta_y = pt2[0] - pt1[0]
        delta_x = pt2[1] - pt1[1]
        angle_deg = np.degrees(
            np.arctan2(delta_y, delta_x)
        )  # https://en.wikipedia.org/wiki/Atan2
        if angle_deg < 0:  # only positive angle
            angle_deg += 180

        return length, angle_deg

    def _analyze_node(self, node_id):
        node_view = self.graph[node_id]
        return {
            "coordinates": self.graph.nodes[node_id]["pts"].flatten().tolist(),
            "num_edges": len(node_view),
            "neighboring_nodes": list(node_view),
        }

    @staticmethod
    def create_edge_id(start_node_id, end_node_id):
        """Edges are defined by start and end nodes. They don't have any ids, so the id must be constructed. Id pattern: LOWERID_HIGHERID"""
        if start_node_id < end_node_id:
            return f"{start_node_id}_{end_node_id}"
        else:
            return f"{end_node_id}_{start_node_id}"

    def analyze_cracks(self):
        """Returns dataframes with node and edge parameters for further analysis."""
        df_nodes = pd.DataFrame(
            columns=["coordinates", "num_edges", "neighboring_nodes"],
            index=pd.Index([], name="node_id"),
        )
        df_edges = pd.DataFrame(
            columns=["num_pts", "start_node", "end_node", "length", "angle"],
            index=pd.Index([], name="edge_id"),
        )
        for start_node_id, end_node_id in self.graph.edges():
            pts = self.graph.get_edge_data(start_node_id, end_node_id)["pts"]
            if pts.shape[0] > self.min_number_of_crack_points:
                # analyze nodes
                df_nodes.loc[start_node_id] = self._analyze_node(start_node_id)
                df_nodes.loc[end_node_id] = self._analyze_node(end_node_id)
                # analyze edges
                edge_id = self.create_edge_id(start_node_id, end_node_id)
                edge_params = {
                    "start_node": start_node_id,
                    "end_node": end_node_id,
                    **self._analyze_edge(pts),
                }
                df_edges.loc[edge_id] = edge_params

        return df_nodes, df_edges


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
            self.model_type = "resnext101_32x8d"
        else:
            self.model_type = model_type

        if model_path is None:
            DownloadModel(str(model))
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

        pass

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
        self.img
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
