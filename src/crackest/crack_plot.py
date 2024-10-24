import cv2
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from skimage.morphology import skeletonize


class CrackPlot:
    """Abstract class."""
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
