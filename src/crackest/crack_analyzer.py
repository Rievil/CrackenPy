import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CrackAnalyzer:
    """Analyze crack patterns in the crack graph."""

    def __init__(self, graph, min_number_of_crack_points: int = 20):
        self.graph = graph
        self.min_number_of_crack_points = min_number_of_crack_points

    def graph_stats(self):
        """Prints a basic graph statistics."""
        print(f"directed: {self.graph.is_directed()}")
        print(f"edges: {len(self.graph.edges)}")
        print(f"nodes: {len(self.graph.nodes)}")

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

    def plot_cracks(self, img_r: np.ndarray, selected_edge_ids: list | None = None):
        """Plot image with nodes and edges (cracks). Specific cracks can be selected by edge_ids."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.imshow(img_r, cmap="gray")

        for start_node_id, end_node_id in self.graph.edges():
            edge_id = self.create_edge_id(start_node_id, end_node_id)
            if selected_edge_ids is not None and edge_id not in selected_edge_ids:
                continue

            pts = self.graph.get_edge_data(start_node_id, end_node_id)["pts"]
            if pts.shape[0] > self.min_number_of_crack_points:
                start_end_pts = pts[[0, -1]]
                # swapped X and Y coordinates?
                yps = start_end_pts.take(0, axis=1)
                xps = start_end_pts.take(1, axis=1)

                ax.scatter(xps, yps, color="red")  # plot start/end nodes
                ax.plot(pts[:, 1], pts[:, 0], "green")  # plot edge

        ax.set_xlim([0, img_r.shape[1]])
        ax.set_ylim([0, img_r.shape[0]])
        ax.axis("off")
        plt.show()
