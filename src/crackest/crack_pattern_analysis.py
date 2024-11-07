import cv2
import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from skimage import measure
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import skeletonize
import sknw


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

        if df_edges["length"].sum() == 0:
            mean_angle_weighted = 0
        else:
            mean_angle_weighted = (
                df_edges["angle"] * df_edges["length"]
            ).sum() / df_edges["length"].sum()

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
