# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 12:32:38 2024

@author: Richard
"""
import cv2
import numpy as np
from skimage import measure
import pandas as pd

from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
import math
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import rotate
from crackest import cracks as cr
import os
import datetime
from tqdm import tqdm


class CrackAn:
    def __init__(self, cp=None):
        if cp is not None:
            self.cracpy = cp
        else:
            self.cracpy = cr.CrackPy(model=0)

        self.img = []
        self.mask = []
        self.masks = []
        self.pixel_mm_ratio = []
        self.shape = []
        self.imformat = "png"
        self.regime = "folder"
        self.hasilist = False

    def __MeasureSpecimen__(self, specmask):
        spec_bw = self._getspec_(specmask)

        r = spec_bw.astype(np.uint8)

        kernel = np.ones((20, 20), np.uint8)
        r = cv2.erode(r, kernel)
        r = cv2.dilate(r, kernel, iterations=1)

        contours = measure.find_contours(r, 0.8)
        centroid = measure.centroid(r)

        # print(contours)
        return contours, centroid

    def _getspec_(self, bw):
        mask = bw
        r = mask[:, :] == 0
        r = ~r
        return r

    def MakeIList(self, folder):
        df = pd.DataFrame()
        for root, dirs, files in os.walk(folder, topdown=False):
            for name in files:
                file_path = root + "/" + name
                created = os.path.getctime(file_path)
                modified = os.path.getmtime(file_path)

                date = datetime.datetime.fromtimestamp(modified)
                dfi = pd.DataFrame(
                    {
                        "folder": root,
                        "name": name,
                        "modified": modified,
                        "date": date,
                    },
                    index=[0],
                )
                df = pd.concat([df, dfi], axis=0)

        extension = df["name"].str.split(".", expand=True)

        idx = extension[1] == self.imformat.lower()

        df = df[idx]

        df["order"] = df["modified"] - df["modified"].min()
        df = df.sort_values(by="order", ascending=True, ignore_index=True)

        t1 = df["date"][0]
        offset = (
            t1.hour / 24 + t1.minute / (24 * 60) + t1.second / (24 * 60 * 60)
        )

        df["order"] = df["order"] / (24 * 3600) + offset

        # df=df.reset_index()
        df["name"] = df["name"].str.lower()

        df["days"] = np.fix(df["order"].values)
        hours = (df["order"].values - np.fix(df["order"].values)) * 24
        minutes = (hours - np.fix(hours)) * 60
        seconds = (minutes - np.fix(minutes)) * 60

        df["hours"] = np.fix(hours)
        df["minutes"] = np.fix(minutes)
        df["seconds"] = (minutes - np.fix(minutes)) * 60

        prog = df["order"]
        prog = prog - prog.min()
        prog = prog / prog.max()

        # prog=prog/prog.max()

        df["progress"] = prog

        self.imglist = df
        self.hasilist = True

    def __rotate_image__(self, mat, angle):
        """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        """

        height, width = mat.shape[:2]  # image shape has 3 dimensions
        image_center = (
            width / 2,
            height / 2,
        )  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

        return rotated_mat

    def Input(self, file=None, folder=None, **kwargs):
        options = {"gamma_correction": 1, "black_level": 1}
        options.update(kwargs)
        self.cracpy_sett = kwargs

        if file is not None:
            self.imfile = file
            self.regime = "file"

        if folder is not None:
            self.folder = folder
            self.regime = "folder"
            self.MakeIList(folder)

    def __stabil_img__(self, img, sett):
        pass

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

        img2 = cv2.rectangle(
            img2, startp, endp, color=(255, 255, 255), thickness=-1
        )
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

        img2 = cv2.rectangle(
            img2, r_startp, r_endp, color=(150, 50, 50), thickness=-1
        )

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

    def ShowRegistr(self, trsh=0.8):
        df = self.specimens
        df = df[df["cover"] > trsh]

        fig, ax = plt.subplots()
        ax.imshow(self.cracpy.img)

        for index, row in df.iterrows():
            sp = row["subspec"]
            bbox = row["bbox"]

            bx = [bbox[1], bbox[3], bbox[3], bbox[1], bbox[1]]
            by = [bbox[0], bbox[0], bbox[2], bbox[2], bbox[0]]
            ax.plot(bx, by, "-w", linewidth=2.5, alpha=0.7)
            ax.text(
                np.mean(bx),
                np.mean(by),
                "Spec. {:d}".format(index),
                color="tab:red",
                fontsize=20,
                verticalalignment="center",
                horizontalalignment="left",
            )

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        fig.tight_layout()
        return fig

    def Registr(self, currimg=0, frame=10):
        if self.regime == "folder":
            if self.hasilist == False:
                print(
                    "Error: Now Ilist created, specify the folder with images to process"
                )
                return
            else:
                self.currimg_index = currimg
                filename = r"{:s}\\{:s}".format(
                    self.imglist["folder"][self.currimg_index],
                    self.imglist["name"][self.currimg_index],
                )
            # self.cracpy
        elif self.regime == "file":
            self.currimg_index = 0
            filename = self.imfile
            pass

        self.cracpy.GetMask(impath=filename, **self.cracpy_sett)
        self.cracpy.SepMasks()

        reg_props = (
            "area",
            "centroid",
            "orientation",
            "axis_major_length",
            "axis_minor_length",
            "bbox",
        )

        length = 160
        width = 40

        bw_mask = self.cracpy.masks["spec"]

        kernel = np.ones((50, 50), np.uint8)
        bw_mask = cv2.dilate(bw_mask, kernel, iterations=1)
        bw_mask = cv2.erode(bw_mask, kernel)

        shape = bw_mask.shape

        image = bw_mask.astype(np.uint8)
        label_img = label(image)
        # regions = regionprops(label_img)

        props_mat = regionprops_table(label_img, properties=reg_props)
        dfmat = pd.DataFrame(props_mat)
        dfmat.sort_values(by=["area"], ascending=True)
        dfmat = dfmat.reset_index()

        # %
        # fig, (ax,ax1,ax3) = plt.subplots(nrows=3,ncols=1)
        # ax.imshow(image, cmap=plt.cm.gray)
        #
        dfii = pd.DataFrame()
        n = 0
        for index, props in dfmat.iterrows():
            if props["area"] > 1000:
                n = n + 1
                y0 = props["centroid-0"]
                x0 = props["centroid-1"]
                orientation = props["orientation"]

                rat1 = 0.43
                x0i = (
                    x0
                    - math.cos(orientation) * rat1 * props["axis_minor_length"]
                )
                y0i = (
                    y0
                    + math.sin(orientation) * rat1 * props["axis_minor_length"]
                )

                x1 = (
                    x0
                    + math.cos(orientation) * rat1 * props["axis_minor_length"]
                )
                y1 = (
                    y0
                    - math.sin(orientation) * rat1 * props["axis_minor_length"]
                )

                rat2 = 0.43
                x2i = (
                    x0
                    + math.sin(orientation) * rat2 * props["axis_major_length"]
                )
                y2i = (
                    y0
                    + math.cos(orientation) * rat2 * props["axis_major_length"]
                )

                x2 = (
                    x0
                    - math.sin(orientation) * rat2 * props["axis_major_length"]
                )
                y2 = (
                    y0
                    - math.cos(orientation) * rat2 * props["axis_major_length"]
                )

                he = {"alpha": (y0i - y1) / (x0i - x1)}
                he["beta"] = y1 - he["alpha"] * x1
                he["len"] = props["axis_minor_length"]
                he["label"] = "width"

                # ax.plot((x2i, x2), (y2i, y2), '-r', linewidth=2.5)

                ve = {"alpha": (y2i - y2) / (x2i - x2)}
                ve["beta"] = y2 - ve["alpha"] * x2
                ve["len"] = props["axis_major_length"]
                ve["label"] = "length"

                # ax.plot(x0, y0, '.g', markersize=15)

                bbox = [
                    int(props["bbox-0"]),
                    int(props["bbox-1"]),
                    int(props["bbox-2"]),
                    int(props["bbox-3"]),
                ]
                self.__check_bbox__(bbox, shape, frame)

                imgr = np.zeros([bbox[2] - bbox[0], bbox[3] - bbox[1], 3])
                maskr = np.zeros([bbox[2] - bbox[0], bbox[3] - bbox[1]])

                img_r = self.cracpy.img[
                    bbox[0] : bbox[2], bbox[1] : bbox[3], :
                ]
                mask_r = self.cracpy.mask[bbox[0] : bbox[2], bbox[1] : bbox[3]]

                rows, cols, channels = imgr.shape

                if rows > cols:
                    rn = cols
                    cn = rows
                else:
                    cn = cols
                    rn = rows
                angle = angle_in_degrees = orientation * (180 / np.pi) + 90

                img_r = self.__rotate_image__(img_r, -angle)
                mask_r = self.__rotate_image__(mask_r, -angle)

                mask_r = mask_r.round(0)

                sbbox = self.__subcrop__(mask_r)

                maskBW = mask_r[:, :] == 0
                maskBW = ~maskBW
                maskBW = maskBW.astype(np.uint8)

                sett = {
                    "ID": n,
                    "angle": -angle,
                    "label": [],
                    "area": props["area"],
                    "length": length,
                    "width": width,
                    "bbox": [bbox],
                }

                eq = {"h": he, "v": ve}  # break
                xin = (eq["h"]["beta"] - eq["v"]["beta"]) / (
                    eq["v"]["alpha"] - eq["h"]["alpha"]
                )
                yin = xin * eq["h"]["alpha"] + eq["h"]["beta"]
                eq["center"] = (xin, yin)

                [contour, speccenter] = self.__MeasureSpecimen__(mask_r)

                spec = {
                    "img": img_r,
                    "mask": mask_r,
                    "contour": contour,
                    "centroid": speccenter,
                }

                dfii = pd.concat(
                    [
                        dfii,
                        pd.DataFrame(
                            {
                                "bbox": [bbox],
                                "sbbox": [sbbox],
                                "angle": angle,
                                "area": props["area"],
                                "x0": x0,
                                "y0": y0,
                                "subspec": [spec],
                                "eq": [eq],
                                "sett": [sett],
                                "ratio": 0,
                            }
                        ),
                    ],
                    ignore_index=True,
                    axis=0,
                )

                dfii = dfii.sort_values(by="area", ascending=False)
                # dfii=dfii.reset_index()

        dfii["cover"] = dfii["area"].values / dfii["area"].max()

        self.specimens = dfii

        return dfii

    def MakeVideo(
        self, spid=0, filename="Output.avi", imrange=None, vidframe=0
    ):
        n = 0

        for index, row in tqdm(
            self.imglist.iterrows(), total=self.imglist.shape[0]
        ):
            imgr = self.GetSubImg(
                index, specid=spid, anotate=True, frame=vidframe
            )
            if n == 0:
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                size = [imgr.shape[1], imgr.shape[0]]
                vidname = filename
                out = cv2.VideoWriter(vidname, fourcc, 8, size)
                n += 1

            out.write(imgr)

        out.release()
        # %

    def __check_bbox__(self, bbox, shape, frame):
        if bbox[0] - frame > 0:
            bbox[0] -= int(frame)
        else:
            bbox[0] = 0

        if bbox[1] + frame < shape[0]:
            bbox[1] += int(frame)
        else:
            bbox[1] = int(shape[0])

        if bbox[2] - frame > 0:
            bbox[2] -= int(frame)
        else:
            bbox[2] = 0

        if bbox[3] + frame < shape[1]:
            bbox[3] += int(frame)
        else:
            bbox[3] = int(shape[1])

        return bbox

    def __subcrop__(self, mask):
        maskBW = mask[:, :] == 0
        maskBW = ~maskBW
        maskBW = maskBW.astype(np.uint8)

        mx = maskBW.sum(axis=0)
        my = maskBW.sum(axis=1)

        mx[mx < 100] = 0
        my[my < 100] = 0

        xlim = np.where(my > 0)[0]
        ylim = np.where(mx > 0)[0]

        self.xylims = [mx, my]

        minx = int(xlim[0])
        maxx = int(xlim[-1])
        miny = int(ylim[0])
        maxy = int(ylim[-1])

        return [minx, maxx, miny, maxy]

    def MakeRatio(self, length=160, width=40):
        mask = self.subspec["mask"]
        bbox = self.__subcrop__(self.cracpy.mask)

        width_px = bbox[1] - bbox[0]
        length_px = bbox[3] - bbox[2]
        if width_px>length_px:
            tmp=length_px
            length_px=width_px
            width_px=tmp

        ratio_l = length_px / length
        ratio_w = width_px / width
        finratio = (ratio_w + ratio_l) / 2
        print(
            "W:{:.5f} L:{:.5f}, MeanR:{:.5f} min dist. {:.3f} um/px".format(
                ratio_w, ratio_l, finratio, 1 / finratio * 1000
            )
        )
        return finratio

    def GetSubImg(
        self,
        imgindex=0,
        specid=0,
        anotate=False,
        prog=None,
        label=None,
        frame=0,
        getmask=False,
    ):
        bbox = self.specimens["bbox"][specid]
        sbox = self.specimens["sbbox"][specid]

        if self.regime == "folder":
            impath = r"{:s}\\{:s}".format(
                self.imglist["folder"][imgindex],
                self.imglist["name"][imgindex],
            )

            self.cracpy.GetImg(impath)

            img = self.cracpy.img.copy()
            imgr = np.zeros([bbox[2] - bbox[0], bbox[3] - bbox[1], 3])
            imgr = img[bbox[0] : bbox[2], bbox[1] : bbox[3], :]

            img_r = self.__rotate_image__(
                imgr, self.specimens["angle"][specid]
            )
            sbox_c = self.__check_bbox__(sbox, img_r.shape, frame)

            img_r = img_r[sbox_c[0] : sbox_c[1], sbox_c[2] : sbox_c[3], :]

            if getmask == True:
                self.cracpy.GetMask(img=img_r)
                mask_r = self.cracpy.mask

        elif self.regime == "file":
            img = self.cracpy.img
            mask = self.cracpy.mask

            imgr = np.zeros([bbox[2] - bbox[0], bbox[3] - bbox[1], 3])
            imgr = img[bbox[0] : bbox[2], bbox[1] : bbox[3], :]

            maskr = np.zeros([bbox[2] - bbox[0], bbox[3] - bbox[1]])
            maskr = mask[bbox[0] : bbox[2], bbox[1] : bbox[3]]

            imgr = self.__rotate_image__(
                imgr, -self.specimens["angle"][specid]
            )
            maskr = self.__rotate_image__(
                maskr, -self.specimens["angle"][specid]
            )

            sboxn = self.__subcrop__(maskr)
            sbox_c = self.__check_bbox__(sboxn, imgr.shape, frame)

            # print(sbox_c)
            # img_r=imgr
            # mask_r=maskr
            img_r = imgr[sbox_c[0] : sbox_c[1], sbox_c[2] : sbox_c[3], :]
            mask_r = maskr[sbox_c[0] : sbox_c[1], sbox_c[2] : sbox_c[3]]

        if label is None:
            if self.regime == "folder":
                label = "{:02.0f}:{:02.0f}".format(
                    self.imglist["days"][imgindex],
                    self.imglist["hours"][imgindex],
                )
            elif self.regime == "file":
                label = "Specimen ID {:d}".format(specid)

        if prog is None:
            if self.regime == "folder":
                prog = self.imglist["progress"][imgindex]
            elif self.regime == "file":
                prog = 0

        self.subspec = {
            "ID": specid,
            "ImgID": imgindex,
            "mask": mask_r,
            "img": img_r,
        }
        self.subspec["ratio"] = self.MakeRatio(length=160, width=40)

        if anotate == True:
            img_r = self.__anotate_img__(img_r, prog, label)

        return img_r, mask_r
        # img_r=rotate(cp.img[minr:maxr,minc:maxc,:], orientation)
        # mask_r=rotate(cp.mask[minr:maxr,minc:maxc], orientation)