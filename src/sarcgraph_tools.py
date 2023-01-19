import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import pickle
import imageio
import shutil
import json

from scipy import signal
from scipy.signal import find_peaks
from scipy.linalg import polar
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

from pathlib import Path

from src.sarcgraph import SarcGraph


class SarcGraphTools:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.output_dir = input_dir
        self.visualization = self.Visualization(self)
        self.time_series = self.TimeSeries(self)
        self.analysis = self.Analysis(self)

    class TimeSeries:
        def __init__(self, sg_tools):
            self.sg_tools = sg_tools

        def normalize(self, data):
            mu = np.mean(data, axis=1, keepdims=True)
            return (data - mu) / mu

        def dtw_distance(self, s1, s2):
            """Compute distance based on dynamic time warping (DTW)"""
            if s1.ndim == 1 and s2.ndim == 1:
                n = len(s1)
                m = len(s2)
                dtw = np.inf * np.ones((n + 1, m + 1))
                dtw[0, 0] = 0
                dist = (s1.reshape(-1, 1) - s2.reshape(1, -1)) ** 2
                for i in range(1, n + 1):
                    for j in range(1, m + 1):
                        dtw[i, j] = dist[i - 1, j - 1] + min(
                            dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1]
                        )
            return np.sqrt(dtw[-1, -1])

        def gpr(self, data):
            num_frames = len(data)

            kernel = 1.0 * RBF(
                length_scale=1.0, length_scale_bounds=(1e-5, 10.0)
            ) + 1.0 * WhiteKernel(
                noise_level=1.0, noise_level_bounds=(1e-5, 10.0)
            )
            model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
            xdata = np.arange(num_frames)
            ydata = data

            remove_indices = np.logical_not(np.isnan(ydata))
            xhat = xdata.reshape(-1, 1)
            xdata = xdata[remove_indices].reshape(-1, 1)
            ydata = ydata[remove_indices].reshape(-1, 1)

            model.fit(xdata, ydata)
            yhat = model.predict(xhat)

            return yhat.reshape(-1)

        def sarc_info_gpr(self, save_data=True):
            # --> use GPR to interpolate sarcomere length for all frames
            sarcs_info = np.load(
                f"{self.sg_tools.input_dir}/sarcomeres-info.npy"
            )

            sarcs_info_gpr = np.zeros(sarcs_info.shape)
            for info_type in range(sarcs_info.shape[0]):
                for particle_num in range(sarcs_info.shape[1]):
                    sarcs_info_gpr[info_type, particle_num, :] = self.gpr(
                        sarcs_info[info_type, particle_num, :]
                    )
            if save_data:
                np.save(
                    f"{self.sg_tools.output_dir}/sarcomeres-info-gpr.npy",
                    sarcs_info_gpr,
                )
            return sarcs_info_gpr

    class Visualization:
        def __init__(self, sg_tools):
            self.sg_tools = sg_tools

        def zdiscs_and_sarcs(self, frame_num=0, include_eps=False):
            """Visualize the results of z-disk segmentation."""
            # load raw image file
            raw_img = np.load(f"{self.sg_tools.input_dir}/raw-frames.npy")[
                frame_num
            ]

            # load segmented zdisc contours
            contour_list = np.load(
                f"{self.sg_tools.input_dir}/contours.npy", allow_pickle=True
            )[frame_num]

            # load sarcomere info
            sarc_data = np.load(
                f"{self.sg_tools.input_dir}/sarcomeres-info.npy"
            )[:, :, frame_num]
            sarc_x = sarc_data[0, :]
            sarc_y = sarc_data[1, :]

            ax = plt.axes()
            ax.set_aspect("equal")
            ax.imshow(raw_img[:, :, 0], cmap=plt.cm.gray)
            ax.set_title(
                f"{len(contour_list)} z-disks and {len(sarc_x)} sarcomeres\n"
                f"found in frame {frame_num}"
            )
            for contour in contour_list:
                ax.plot(contour[:, 1], contour[:, 0], "#0000cc", linewidth=1)
            ax.plot(sarc_y, sarc_x, "*", color="#cc0000", markersize=3)
            ax.set_xticks([])
            ax.set_yticks([])

            output_file = (
                f"{self.sg_tools.output_dir}/zdiscs-sarcs-frame-{frame_num}"
            )
            Path(f"{self.sg_tools.output_dir}").mkdir(
                parents=True, exist_ok=True
            )
            plt.savefig(f"{output_file}.png", dpi=300, bbox_inches="tight")
            if include_eps:
                plt.savefig(f"{output_file}.eps", bbox_inches="tight")

        def contraction(self):
            "Visualize detected and tracked sarcomeres."
            try:
                sarcs_data = np.load(
                    f"{self.sg_tools.output_dir}/sarcomeres-info-gpr.npy"
                )
            except FileNotFoundError:
                sarcs_data = self.sg_tools.time_series.sarc_info_gpr(
                    save_data=False
                )

            sarcs_x = sarcs_data[0]
            sarcs_y = sarcs_data[1]
            sarcs_length = sarcs_data[2]
            num_frames = sarcs_x.shape[-1]

            sarcs_length_mean = np.nanmean(sarcs_length, axis=1, keepdims=True)
            sarcs_length_norm = (
                sarcs_length - sarcs_length_mean
            ) / sarcs_length_mean
            # --> plot every frame, plot every sarcomere according to normalized
            # fraction length
            color_matrix = np.less(np.abs(sarcs_length_norm), 0.2) * (
                sarcs_length_norm * 2.5 + 0.5
            ) + np.greater(sarcs_length_norm, 0.2)

            img_list = []
            Path("tmp").mkdir(parents=True, exist_ok=True)
            for frame_num in range(num_frames):
                raw_img = np.load(f"{self.sg_tools.input_dir}/raw-frames.npy")[
                    frame_num, :, :, 0
                ]

                plt.figure()
                plt.imshow(raw_img, cmap=plt.cm.gray)
                num_sarcs = sarcs_length_norm.shape[0]
                for sarc_num in range(num_sarcs):
                    c = color_matrix[sarc_num, frame_num]
                    col = (1 - c, 0, c)
                    y = sarcs_x[sarc_num, frame_num]
                    x = sarcs_y[sarc_num, frame_num]
                    plt.scatter(x, y, s=15, color=col, marker="o")

                ax = plt.gca()
                ax.set_xticks([])
                ax.set_yticks([])
                plt.savefig(
                    f"tmp/frame-{frame_num}.png", dpi=300, bbox_inches="tight"
                )
                plt.close()
                img_list.append(imageio.imread(f"tmp/frame-{frame_num}.png"))
            shutil.rmtree("tmp")
            if num_frames > 1:
                imageio.mimsave(
                    f"{self.sg_tools.output_dir}/contract_anim.gif", img_list
                )
            print(
                f"GIF saved as '{self.sg_tools.output_dir}/contract_anim.gif'!"
            )

        def plot_normalized_sarcs_length(self, quality=100, include_eps=False):
            try:
                sarcs_data = np.load(
                    f"{self.sg_tools.output_dir}/sarcomeres-info-gpr.npy"
                )
            except FileNotFoundError:
                sarcs_data = self.sg_tools.timeseries.sarc_info_gpr(
                    save_data=False
                )

            sarcs_length_norm = self.sg_tools.time_series.normalize(
                sarcs_data[2]
            )
            num_sarcs = sarcs_length_norm.shape[0]

            plt.plot(sarcs_length_norm.T, linewidth=0.25)
            plt.plot(
                np.median(sarcs_length_norm, axis=0),
                "k-",
                linewidth=3,
                label="median curve",
            )
            plt.plot(
                np.mean(sarcs_length_norm, axis=0),
                "--",
                color=(0.5, 0.5, 0.5),
                linewidth=3,
                label="mean curve",
            )
            plt.xlabel("frame")
            plt.ylabel("normalized length")
            plt.title(
                f"timeseries data, tracked and normalized, {num_sarcs} sarcomeres"
            )
            plt.ylim((-0.1, 0.1))
            plt.legend()
            plt.tight_layout()

            plt.savefig(
                f"{self.sg_tools.output_dir}/normalized_sarcomeres_length_plot.png",
                dpi=quality,
                bbox_inches="tight",
            )
            if include_eps:
                plt.savefig(
                    f"{self.sg_tools.output_dir}/normalized_sarcomeres_length_plot.eps",
                    bbox_inches="tight",
                )

        def plot_OOP(self, quality=100, include_eps=False):
            try:
                OOP = np.load(f"{self.sg_tools.output_dir}/recovered_OOP.npy")
            except FileNotFoundError:
                OOP, _ = self.sg_tools.analysis.compute_OOP(save_data=False)

            plt.figure(figsize=(5, 5))
            plt.subplot(1, 1, 1)
            plt.plot(OOP, "k-", label="OOP recovered")
            plt.legend()
            plt.title("recovered Orientational Order Parameter")
            plt.xlabel("frames")

            plt.savefig(
                f"{self.sg_tools.output_dir}/recovered_OOP_plot.png",
                dpi=quality,
                bbox_inches="tight",
            )
            if include_eps:
                plt.savefig(
                    f"{self.sg_tools.output_dir}/recovered_OOP_plot.eps",
                    bbox_inches="tight",
                )

        def plot_F(self, include_eps=False):
            try:
                F = np.load(f"{self.sg_tools.output_dir}/recovered_F.npy")
            except FileNotFoundError:
                F, _ = self.sg_tools.analysis.compute_F(save_data=False)

            plt.figure(figsize=(5, 5))
            plt.subplot(1, 1, 1)
            plt.plot(F[:, 0, 0] - 1, "r--", linewidth=5, label="F11 recovered")
            plt.plot(F[:, 1, 1] - 1, "g--", linewidth=4, label="F22 recovered")
            plt.plot(F[:, 0, 1], "c:", label="F12 recovered")
            plt.plot(F[:, 1, 0], "b:", label="F21 recovered")
            plt.legend()
            plt.title("recovered deformation gradient")
            plt.xlabel("frames")
            plt.savefig(
                f"{self.sg_tools.output_dir}/recovered_F_plot.png",
                dpi=300,
                bbox_inches="tight",
            )
            if include_eps:
                plt.savefig(
                    f"{self.sg_tools.output_dir}/recovered_F_plot.eps",
                    bbox_inches="tight",
                )

        def plot_J(self, include_eps=False):
            """Analyze the Jacobian -- report timeseries parmeters. Must first run
            compute_F_whole_movie()."""
            try:
                J = np.load(f"{self.sg_tools.output_dir}/recovered_J.npy")
            except FileNotFoundError:
                _, J = self.sg_tools.analysis.compute_F(save_data=False)
            frames = np.arange(len(J))

            # compute the parameters of the timeseries
            plt.figure(figsize=(5, 5))
            plt.plot(J, "k-")

            J_med = signal.medfilt(J, 5)
            J_deriv = np.gradient(J, frames)
            count_C = 0
            count_R = 0
            count_F = 0
            thresh_flat = 0.01 * (np.max(J) - np.min(J))

            for frame_num in range(len(frames)):
                if J_deriv[frame_num] > thresh_flat:
                    count_R += 1
                    plt.plot(
                        frames[frame_num],
                        J[frame_num],
                        "o",
                        color=(0.5, 0.5, 0.5),
                    )
                elif J_deriv[frame_num] < -1.0 * thresh_flat:
                    count_C += 1
                    plt.plot(
                        frames[frame_num], J[frame_num], "o", color=(0.5, 0, 0)
                    )
                else:
                    count_F += 1
                    plt.plot(
                        frames[frame_num], J[frame_num], "o", color=(0, 0, 0.5)
                    )

            # detect peaks and valleys
            # peaks_U, _ = find_peaks(data_med, threshold=th, distance=di, width=wi)
            peaks_L, _ = find_peaks(
                -1.0 * J_med, threshold=0.0, distance=10, width=5
            )
            plt.grid()
            # plt.plot(x[peaks_U],data[peaks_U],'rx',markersize=10)
            plt.plot(frames[peaks_L], J[peaks_L], "rx", markersize=13)
            plt.title(
                "frames contract: %i, relax: %i, flat: %i"
                % (count_C, count_R, count_F)
            )
            plt.xlabel("frame number")
            plt.ylabel("determinate of average F")
            plt.tight_layout()

            plt.savefig(
                f"{self.sg_tools.output_dir}/recovered_J_plot.png",
                dpi=300,
                bbox_inches="tight",
            )
            if include_eps:
                plt.savefig(
                    f"{self.sg_tools.output_dir}/recovered_J_plot.eps",
                    bbox_inches="tight",
                )
            return

        def F_eigenval_animation(
            self, quality=100, include_eps=False, save_data=True
        ):
            """Visualize the eigenvalues of F -- plot timeseries next to the movie.
            Must first run compute_F_whole_movie()."""
            try:
                F_all = np.load(f"{self.sg_tools.output_dir}/recovered_F.npy")
                J_all = np.load(f"{self.sg_tools.output_dir}/recovered_J.npy")
            except FileNotFoundError:
                F_all, J_all = self.sg_tools.analysis.compute_F(
                    save_data=False
                )

            try:
                OOP_all = np.load(
                    f"{self.sg_tools.output_dir}/recovered_OOP.npy"
                )
                OOP_vec_all = np.load(
                    f"{self.sg_tools.output_dir}/recovered_OOP_vec.npy"
                )
            except FileNotFoundError:
                OOP_all, OOP_vec_all = self.sg_tools.analysis.compute_OOP(
                    save_data=False
                )

            num_frames = len(J_all)
            frames = np.arange(num_frames)
            raw_imgs = np.load(f"{self.sg_tools.input_dir}/raw-frames.npy")[
                :, :, :, 0
            ]

            R_all = np.zeros((num_frames, 2, 2))
            U_all = np.zeros((num_frames, 2, 2))
            lambda_1 = np.zeros(num_frames)
            lambda_2 = np.zeros(num_frames)
            vec_1 = np.zeros((num_frames, 2))
            vec_2 = np.zeros((num_frames, 2))
            for frame_num, f in enumerate(F_all):
                R, U = polar(f)
                R_all[frame_num] = R
                U_all[frame_num] = U
                w, v = np.linalg.eig(U)
                lambda_1[frame_num] = np.min(w)
                lambda_2[frame_num] = np.max(w)
                v = R.dot(v)
                vec_1[frame_num] = v[:, np.argmin(w)]
                vec_2[frame_num] = v[:, np.argmax(w)]

            img_list = []
            Path("tmp").mkdir(parents=True, exist_ok=True)
            radius = 0.2 * np.min(raw_imgs.shape[1:])
            th = np.linspace(0, 2.0 * np.pi, 100)
            v = np.array([radius * np.cos(th), radius * np.sin(th)]).T
            center = np.array(raw_imgs.shape[1:]).reshape(1, 2) / 2
            vec_circ = v + center
            p1_1 = center - radius * vec_1
            p1_2 = center + radius * vec_1
            p2_1 = center - radius * vec_2
            p2_2 = center + radius * vec_2
            for frame_num, raw_img in enumerate(raw_imgs):
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(raw_img, cmap=plt.cm.gray)
                plt.plot(
                    [p1_1[frame_num][1], p1_2[frame_num][1]],
                    [p1_1[frame_num][0], p1_2[frame_num][0]],
                    "-",
                    color=(255 / 255, 204 / 255, 203 / 255),
                    linewidth=0.3,
                )
                plt.plot(
                    [p2_1[frame_num][1], p2_2[frame_num][1]],
                    [p2_1[frame_num][0], p2_2[frame_num][0]],
                    "-",
                    color=(0.5, 0.5, 0.5),
                    linewidth=0.3,
                )
                # add in eigenvector directions
                f = F_all[frame_num]
                v_def = v.dot(np.linalg.matrix_power(f, 9))
                vec_inner_circ = v_def + center

                plt.plot(
                    vec_circ[:, 1],
                    vec_circ[:, 0],
                    "-",
                    color=(255 / 255, 204 / 255, 203 / 255),
                    linewidth=0.3,
                )
                plt.plot(
                    vec_inner_circ[:, 1],
                    vec_inner_circ[:, 0],
                    "-",
                    color=(255 / 255, 204 / 255, 203 / 255),
                    linewidth=1.0,
                )

                OOP_vec = OOP_vec_all[frame_num]
                OOP_rad = radius * OOP_all[frame_num]

                plt.plot(
                    [
                        center[1] - OOP_vec[1] * OOP_rad,
                        center[1] + OOP_vec[1] * OOP_rad,
                    ],
                    [
                        center[0] - OOP_vec[0] * OOP_rad,
                        center[0] + OOP_vec[0] * OOP_rad,
                    ],
                    "r-",
                    linewidth=5,
                )

                ax = plt.gca()
                ax.set_xticks([])
                ax.set_yticks([])

                plt.subplot(1, 2, 2)
                plt.plot(
                    frames, lambda_1, "-", color="k", linewidth=1, label="λ1"
                )
                plt.plot(
                    frames,
                    lambda_2,
                    "-",
                    color="#7F7F7F",
                    linewidth=1,
                    label="λ2",
                )
                plt.plot(
                    frame_num,
                    lambda_1[frame_num],
                    "o",
                    mfc="#B30000",
                    mec="k",
                    markersize=7,
                )
                plt.plot(
                    frame_num,
                    lambda_2[frame_num],
                    "o",
                    mfc="#B30000",
                    mec="#7F7F7F",
                    markersize=7,
                )
                plt.xlabel("frame number")
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    f"tmp/frame-{frame_num}.png",
                    dpi=quality,
                    bbox_inches="tight",
                )
                plt.close()
                img_list.append(imageio.imread(f"tmp/frame-{frame_num}.png"))
            shutil.rmtree("tmp")

            if save_data:
                with open(
                    f"{self.sg_tools.output_dir}/recovered_lambda.npy", "wb"
                ) as file:
                    np.save(file, np.array([lambda_1, lambda_2]))
            if num_frames > 1:
                imageio.mimsave(
                    f"{self.sg_tools.output_dir}/F_anim.gif", img_list
                )

        def timeseries_params(
            self, quality=100, include_eps=False, save_data=True
        ):
            try:
                data = pd.read_pickle(
                    f"./{self.sg_tools.output_dir}/timeseries_params.pkl"
                )
            except FileNotFoundError:
                data = self.sg_tools.analysis.compute_ts_params(
                    save_data=False
                )

            plt.figure(figsize=(7, 7))

            med = np.median(data["mean_contract_time"])
            plt.subplot(2, 2, 1)
            plt.hist(data["mean_contract_time"])
            plt.plot([med, med], [0, 10], "r--")
            plt.xlabel("frames")
            plt.title(f"median_contract: {med:.2f}")
            plt.tight_layout()

            med = np.median(data["mean_relax_time"])
            plt.subplot(2, 2, 2)
            plt.hist(data["mean_relax_time"])
            plt.plot([med, med], [0, 10], "r--")
            plt.xlabel("frames")
            plt.title(f"median_relax: {med:.2f}")
            plt.tight_layout()

            med = np.median(data["mean_flat_time"])
            plt.subplot(2, 2, 3)
            plt.hist(data["mean_flat_time"])
            plt.plot([med, med], [0, 10], "r--")
            plt.xlabel("frames")
            plt.title(f"median_flat: {med:.2f}")
            plt.tight_layout()

            med = np.median(data["mean_period_len"])
            plt.subplot(2, 2, 4)
            plt.hist(data["mean_period_len"])
            plt.plot([med, med], [0, 10], "r--")
            plt.xlabel("frames")
            plt.title(f"median_period: {med:.2f}")
            plt.tight_layout()

            plt.savefig(
                f"./{self.sg_tools.output_dir}/histogram_time_constants",
                dpi=quality,
                bbox_inches="tight",
            )
            if include_eps:
                plt.savefig(
                    f"./{self.sg_tools.output_dir}//histogram_time_constants.eps"
                )

        def dendrogram(self, dist_func="dtw"):
            """Cluster timeseries and plot a dendrogram that shows the clustering."""
            try:
                sarcs_data = np.load(
                    f"{self.sg_tools.output_dir}/sarcomeres-info-gpr.npy"
                )
            except FileNotFoundError:
                sarcs_data = self.sg_tools.time_series.sarc_info_gpr(
                    save_data=False
                )
            sarcs_length = self.sg_tools.time_series.normalize(sarcs_data[2])
            num_sarcs = len(sarcs_length)

            if dist_func == "dtw":
                dtw_dist = self.sg_tools.time_series.dtw_distance
                dist_mat = np.zeros((num_sarcs, num_sarcs))
                for sarc_1_id in range(num_sarcs):
                    for sarc_2_id in range(sarc_1_id + 1, num_sarcs):
                        dist = dtw_dist(
                            sarcs_length[sarc_1_id, :],
                            sarcs_length[sarc_2_id, :],
                        )
                        dist_mat[sarc_1_id, sarc_2_id] = dist
                        dist_mat[sarc_2_id, sarc_1_id] = dist
            if dist_func == "euclidean":
                dist_mat = squareform(pdist(sarcs_length, "euclidean"))

            dist_v = squareform(dist_mat)
            Z = linkage(dist_v, method="ward", metric="euclidean")

            # --> plot dendrogram
            plt.figure(figsize=(9, 30), frameon=False)
            plt.subplot(1, 2, 1)
            # dendrogram
            dn1 = dendrogram(
                Z,
                orientation="left",
                color_threshold=0,
                above_threshold_color="k",
            )
            ordered = dn1["leaves"]  # from bottom to top

            ax = plt.gca()
            ax.xaxis.set_visible(False)
            plt.subplot(1, 2, 2)
            ax = plt.gca()

            for kk in range(len(ordered)):
                ix = ordered[kk]
                col = (
                    1 - kk / len(ordered),
                    kk / len(ordered),
                    1 - kk / len(ordered),
                )
                plt.plot(sarcs_length[ix, :] + kk * 0.3, c=col)

            plt.tight_layout()
            plt.ylim((-0.4, kk * 0.3 + 0.35))
            plt.axis("off")

            plt.savefig(f"{self.sg_tools.output_dir}/dendrogram_DTW.pdf")

        def spatial_graph(self, quality=100, include_eps=False):
            G = nx.read_gpickle(
                f"{self.sg_tools.output_dir}/spatial-graph.pkl"
            )

            with open(
                f"{self.sg_tools.output_dir}/spatial-graph-pos.pkl", "rb"
            ) as file:
                pos = pickle.load(file)

            for node_1, node_2 in G.edges:
                pos_1 = np.array(
                    [G.nodes[node_1]["y_pos"], -G.nodes[node_1]["x_pos"]]
                )
                pos_2 = np.array(
                    [G.nodes[node_2]["y_pos"], -G.nodes[node_2]["x_pos"]]
                )
                ang = 1 / np.sqrt(
                    np.sum(((pos_1 - pos_2) / (pos_1[0] - pos_2[0])) ** 2)
                )
                G[node_1][node_2]["weight"] = ang

            node_scores = []
            for node in G.nodes:
                edge_list = list(G.edges(node))
                counter = 0
                value = 0
                for i in range(len(edge_list)):
                    node_1 = edge_list[i][0]
                    node_2 = edge_list[i][1]

                    pos_1 = np.array(
                        [G.nodes[node_1]["y_pos"], -G.nodes[node_1]["x_pos"]]
                    )
                    pos_2 = np.array(
                        [G.nodes[node_2]["y_pos"], -G.nodes[node_2]["x_pos"]]
                    )

                    vec_1 = (pos_1 - pos_2) / np.linalg.norm(pos_1 - pos_2, 2)
                    for j in range(i + 1, len(edge_list)):
                        node_1 = edge_list[j][0]
                        node_2 = edge_list[j][1]

                        pos_1 = np.array(
                            [
                                G.nodes[node_1]["y_pos"],
                                -G.nodes[node_1]["x_pos"],
                            ]
                        )
                        pos_2 = np.array(
                            [
                                G.nodes[node_2]["y_pos"],
                                -G.nodes[node_2]["x_pos"],
                            ]
                        )

                        vec_2 = (pos_1 - pos_2) / np.linalg.norm(
                            pos_1 - pos_2, 2
                        )

                        value += np.abs(np.dot(vec_1, vec_2))
                        counter += 1

                if counter:
                    node_scores.append(value / counter)
                else:
                    node_scores.append(0)

            plt.figure(figsize=(5, 5))
            edges, weights = zip(*nx.get_edge_attributes(G, "weight").items())
            nx.draw(
                G,
                pos,
                node_color="k",
                node_size=10,
                width=2,
                edge_color=weights,
                edge_cmap=plt.cm.rainbow,
            )

            mi = np.min(node_scores)
            ma = np.max(node_scores)
            for node, node_score in zip(G.nodes, node_scores):
                pos = np.array(
                    [G.nodes[node]["y_pos"], -G.nodes[node]["x_pos"]]
                )
                color_val = 1 - (0.75 * (node_score - mi) / (ma - mi) + 0.25)
                color = (color_val, color_val, color_val)
                if node_score > 0.9:
                    plt.plot(pos[0], pos[1], ".", color=color, ms=10)
                if node_score > 0.75:
                    plt.plot(pos[0], pos[1], ".", color=color, ms=7.5)
                else:
                    plt.plot(pos[0], pos[1], ".", color=color, ms=5)

            plt.savefig(
                f"./{self.sg_tools.output_dir}/spatial-graph.png",
                dpi=quality,
                bbox_inches="tight",
            )
            if include_eps:
                plt.savefig(f"./{self.sg_tools.output_dir}/spatial-graph.eps")

        def tracked_vs_untracked(
            self,
            start_frame=0,
            stop_frame=np.inf,
            quality=100,
            include_eps=False,
        ):
            # process the whole video and detect and track sarcomeres
            sarcs_info = np.load(
                f"{self.sg_tools.output_dir}/sarcomeres-info.npy"
            )

            stop_frame = min(stop_frame, sarcs_info.shape[-1])
            start_frame = min(start_frame, stop_frame - 1)
            num_frames = stop_frame - start_frame

            sarcs_info = sarcs_info[:, :, start_frame:stop_frame]

            # process the video frame by frame and detect sarcomeres without tracking
            sg_video = SarcGraph("test-output", "video")
            segmented_zdiscs = sg_video.zdisc_segmentation(
                "samples/sample_0.avi"
            )

            length_all_frames = []
            width_all_frames = []
            angle_all_frames = []
            median_length_all_frames = []
            sarc_num_all_frames = []
            sg_image = SarcGraph("test-output", "image")
            for frame in range(start_frame, stop_frame):
                segmented_zdiscs_frame = segmented_zdiscs.loc[
                    segmented_zdiscs.frame == frame
                ].copy()
                segmented_zdiscs_frame.loc[:, "frame"] = 0.0
                tracked_zdiscs_frame = sg_image.zdisc_tracking(
                    zdiscs_info=segmented_zdiscs_frame
                )
                _, sarc_info = sg_image.sarcomere_detection(
                    tracked_zdiscs=tracked_zdiscs_frame
                )
                length_all_frames.append(sarc_info[2, :, 0])
                width_all_frames.append(sarc_info[3, :, 0])
                angle_all_frames.append(sarc_info[4, :, 0])
                median_length_all_frames.append(
                    np.median(length_all_frames[-1])
                )
                sarc_num_all_frames.append(sarc_info.shape[1])

            # compute average number of not tracked sarcomeres in each frame
            num_tracked = sarcs_info.shape[1]
            num_not_tracked = np.mean(sarc_num_all_frames)

            len_diff_mean = []
            for untracked_len, tracked_len in zip(
                length_all_frames, sarcs_info[2].T
            ):
                tracked_len_mean = np.nanmean(tracked_len)
                tracked_num = len(tracked_len) - np.sum(np.isnan(tracked_len))
                len_diff_mean.append(
                    self.sg_tools.analysis._sampler(
                        untracked_len,
                        tracked_len_mean,
                        tracked_num,
                        num_run=1000,
                    )
                )

            plt.figure(figsize=(np.clip(int(num_frames * 0.3), 10, 25), 5))
            plt.boxplot(len_diff_mean)
            plt.plot([0, num_frames], [-0.5, -0.5], "k--")
            plt.plot([0, num_frames], [0.5, 0.5], "k--")
            plt.title(
                f"Comparison of length in pixels, approx {num_not_tracked:.2f}\
untracked, {num_tracked} tracked"
            )
            plt.xlabel("frame number")
            plt.ylabel(r"$\mu_{track}-\mu_{all}$")
            plt.savefig(
                f"./{self.sg_tools.output_dir}/length-comparison.png",
                dpi=quality,
                bbox_inches="tight",
            )
            if include_eps:
                plt.savefig(
                    f"./{self.sg_tools.output_dir}/length-comparison.eps"
                )

            wid_diff_mean = []
            for untracked_wid, tracked_wid in zip(
                width_all_frames, sarcs_info[3].T
            ):
                tracked_wid_mean = np.nanmean(tracked_wid)
                tracked_num = len(tracked_wid) - np.sum(np.isnan(tracked_wid))
                wid_diff_mean.append(
                    self.sg_tools.analysis._sampler(
                        untracked_wid,
                        tracked_wid_mean,
                        tracked_num,
                        num_run=1000,
                    )
                )

            plt.figure(figsize=(np.clip(int(num_frames * 0.3), 10, 25), 5))
            plt.boxplot(wid_diff_mean)
            plt.plot([0, num_frames], [-0.5, -0.5], "k--")
            plt.plot([0, num_frames], [0.5, 0.5], "k--")
            plt.title(
                f"Comparison of Width in pixels, approx {num_not_tracked:.2f}\
untracked, {num_tracked} tracked"
            )
            plt.xlabel("frame number")
            plt.ylabel(r"$\mu_{track}-\mu_{all}$")
            plt.savefig(
                f"./{self.sg_tools.output_dir}/width-comparison.png",
                dpi=quality,
                bbox_inches="tight",
            )
            if include_eps:
                plt.savefig(
                    f"./{self.sg_tools.output_dir}/width-comparison.eps"
                )

            ang_diff_mean = []
            rad_diff_mean = []
            for untracked_ang, tracked_ang in zip(
                angle_all_frames, sarcs_info[4].T
            ):
                (
                    tracked_ang_mean,
                    tracked_rad_mean,
                ) = self.sg_tools.analysis._angular_mean(tracked_ang)
                tracked_num = len(tracked_ang) - np.sum(np.isnan(tracked_ang))
                ang_diff, rad_diff = self.sg_tools.analysis._angular_sampler(
                    untracked_ang,
                    tracked_ang_mean,
                    tracked_rad_mean,
                    tracked_num,
                    num_run=1000,
                )
                ang_diff_mean.append(ang_diff)
                rad_diff_mean.append(rad_diff)

            plt.figure(figsize=(np.clip(int(num_frames * 0.3), 10, 25), 10))
            plt.subplot(2, 1, 1)
            plt.boxplot(ang_diff_mean)
            plt.plot([0, num_frames], [-np.pi / 8, -np.pi / 8], "k--")
            plt.plot([0, num_frames], [np.pi / 8, np.pi / 8], "k--")
            plt.title(
                f"Comparison of angle in radians, approx {num_not_tracked:.2f}\
untracked, {num_tracked} tracked"
            )
            plt.xlabel("frame number")
            plt.ylabel(r"$\mu_{track}-\mu_{all}$")
            plt.subplot(2, 1, 2)
            plt.boxplot(rad_diff_mean)
            plt.plot([0, num_frames], [0, 0], "r--", label="uniform")
            plt.plot([0, num_frames], [1, 1], "k--", label="oriented")
            plt.title(
                f"Comparison of angle radius in pixels, approx {num_not_tracked:.2f}\
untracked, {num_tracked} tracked"
            )
            plt.xlabel("frame number")
            plt.ylabel(r"$\mu_{track}-\mu_{all}$")
            plt.legend()
            plt.savefig(
                f"./{self.sg_tools.output_dir}/angle-comparison.png",
                dpi=quality,
                bbox_inches="tight",
            )
            if include_eps:
                plt.savefig(
                    f"./{self.sg_tools.output_dir}/angle-comparison.eps"
                )

    class Analysis:
        def __init__(self, sg_tools):
            self.sg_tools = sg_tools

        def _sampler(self, data, mu, tracked_num, num_run=1000):
            samples = np.zeros(num_run)
            for run in range(num_run):
                ids = np.random.randint(0, len(data), size=(tracked_num))
                samples[run] = mu - np.mean(data[ids])
            return samples

        def _angular_mean(self, data):
            x_mean = np.nanmean(np.cos(data))
            y_mean = np.nanmean(np.sin(data))

            mean_angle = np.arctan2(y_mean, x_mean)
            mean_rad = np.linalg.norm([x_mean, y_mean], 2)

            return mean_angle, mean_rad

        def _angular_sampler(
            self, data, mu_ang, mu_rad, tracked_num, num_run=1000
        ):
            ang_samples = np.zeros(num_run)
            rad_samples = np.zeros(num_run)
            for run in range(num_run):
                ids = np.random.randint(0, len(data), size=(tracked_num))
                ang_mean, rad_mean = self.sg_tools.analysis._angular_mean(
                    data[ids]
                )
                ang_samples[run] = mu_ang - ang_mean
                rad_samples[run] = mu_rad - rad_mean

            return ang_samples, rad_samples

        def compute_F(self, adjust_reference=False, save_data=True):
            """Compute the average deformation gradient for the whole movie."""
            try:
                sarcs_data = np.load(
                    f"{self.sg_tools.output_dir}/sarcomeres-info-gpr.npy"
                )
            except FileNotFoundError:
                sarcs_data = self.sg_tools.timeseries.sarc_info_gpr(
                    save_data=False
                )

            sarcs_data = np.load(
                f"{self.sg_tools.output_dir}/sarcomeres-info-gpr.npy"
            )
            sarcs_x = sarcs_data[0]
            sarcs_y = sarcs_data[1]

            # compute Lambda from x_pos and y_pos
            num_sarcs, num_frames = sarcs_x.shape
            num_vecs = int((num_sarcs * num_sarcs - num_sarcs) / 2.0)

            Lambda_list = []
            for frame_num in range(num_frames):
                x_vec = sarcs_x[:, frame_num]
                y_vec = sarcs_y[:, frame_num]

                Lambda = np.zeros((2, num_vecs))
                x_vec_tile = np.tile(x_vec, (num_sarcs, 1))
                v_x = x_vec_tile.T - x_vec_tile
                y_vec_tile = np.tile(y_vec, (num_sarcs, 1))
                v_y = y_vec_tile.T - y_vec_tile

                indices = np.triu_indices(num_sarcs, 1)
                Lambda[0, :] = v_x[indices]
                Lambda[1, :] = v_y[indices]

                Lambda_list.append(Lambda)

            F_all = np.zeros((num_frames, 2, 2))
            J_all = np.zeros(num_frames)
            num_iter = 2 if adjust_reference else 1
            for iter in range(num_iter):
                ref_frame = np.argmax(J_all)
                for target_frame in range(num_frames):
                    Lambda_i = Lambda_list[ref_frame]
                    Lambda_t = Lambda_list[target_frame]
                    term_1 = np.dot(Lambda_t, Lambda_i.T)
                    term_2 = np.linalg.inv(np.dot(Lambda_i, Lambda_i.T))
                    F = np.dot(term_1, term_2)
                    F_all[target_frame] = F
                    J_all[target_frame] = np.linalg.det(F)

            if save_data:
                np.save(f"{self.sg_tools.output_dir}/recovered_F.npy", F_all)
                np.save(f"{self.sg_tools.output_dir}/recovered_J.npy", J_all)

            return F_all, J_all

        def compute_OOP(self, save_data=True):
            # compute_OOP_all
            try:
                sarcs_data = np.load(
                    f"{self.sg_tools.output_dir}/sarcomeres-info-gpr.npy"
                )
            except FileNotFoundError:
                sarcs_data = self.sg_tools.timeseries.sarc_info_gpr(
                    save_data=False
                )

            num_frames = sarcs_data.shape[-1]
            ang = sarcs_data[4]
            n = np.array(
                [
                    [np.cos(ang) ** 2, np.cos(ang) * np.sin(ang)],
                    [np.cos(ang) * np.sin(ang), np.sin(ang) ** 2],
                ]
            )
            t = 2 * n - np.array([[1, 0], [0, 1]]).reshape(2, 2, 1, 1)
            t = np.mean(t, axis=2)

            OOP_all = np.zeros(num_frames)
            OOP_vec_all = np.zeros((num_frames, 2))
            for frame in range(num_frames):
                u, v = np.linalg.eig(t[:, :, frame])
                OOP_all[frame] = np.max(u)
                OOP_vec_all[frame, :] = v[:, np.argmax(u)]

            if save_data:
                np.save(
                    f"{self.sg_tools.output_dir}/recovered_OOP.npy", OOP_all
                )
                np.save(
                    f"{self.sg_tools.output_dir}/recovered_OOP_vector.npy",
                    OOP_vec_all,
                )

            return OOP_all, OOP_vec_all

        def compue_metrics(self, save_data=True):
            try:
                F_all = np.load(f"{self.sg_tools.output_dir}/recovered_F.npy")
                J_all = np.load(f"{self.sg_tools.output_dir}/recovered_J.npy")
            except FileNotFoundError:
                F_all, J_all = self.sg_tools.analysis.compute_F(
                    save_data=False
                )

            try:
                OOP_all = np.load(
                    f"{self.sg_tools.output_dir}/recovered_OOP.npy"
                )
                OOP_vec_all = np.load(
                    f"{self.sg_tools.output_dir}/recovered_OOP_vec.npy"
                )
            except FileNotFoundError:
                OOP_all, OOP_vec_all = self.sg_tools.analysis.compute_OOP(
                    save_data=False
                )
            try:
                sarcs_data = np.load(
                    f"{self.sg_tools.output_dir}/sarcomeres-info-gpr.npy"
                )
            except FileNotFoundError:
                sarcs_data = self.sg_tools.timeseries.sarc_info_gpr(
                    save_data=False
                )

            max_contract_frame = np.argmin(J_all)
            OOP = OOP_all[max_contract_frame]
            OOP_vec = OOP_vec_all[max_contract_frame]
            F = F_all[max_contract_frame]
            J = J_all[max_contract_frame]

            avg_contract = 1 - np.sqrt(J)

            v = OOP_vec
            v_abs = np.linalg.norm(v, 2)
            v0 = np.dot(np.linalg.inv(F), v)
            v0_abs = np.linalg.norm(v0, 2)
            avg_aligned_contract = (v0_abs - v_abs) / v0_abs

            sarcs_length_norm = self.sg_tools.time_series.normalize(
                sarcs_data[2]
            )

            y_max = np.max(sarcs_length_norm, axis=1)
            y_min = np.min(sarcs_length_norm, axis=1)

            s = (y_max - y_min) / (y_max + 1)
            s_til = np.median(s)

            y_avg = np.mean(sarcs_length_norm, axis=0)
            s_avg = (np.max(y_avg) - np.min(y_avg)) / (np.max(y_avg) + 1)

            info_dict = {
                "OOP": OOP,
                "C_iso": avg_contract,
                "C_OOP": avg_aligned_contract,
                "s_til": s_til,
                "s_avg": s_avg,
            }

            if save_data:
                with open(
                    f"{self.sg_tools.output_dir}/recovered_metrics.json", "w"
                ) as file:
                    json.dump(info_dict, file)

            return info_dict

        def compute_ts_params(self, save_data=True):
            """Compute and save timeseries time constants (contraction time, relaxation\
                time, flat time, period, offset, etc.)."""
            try:
                sarcs_data = np.load(
                    f"{self.sg_tools.output_dir}/sarcomeres-info-gpr.npy"
                )
            except FileNotFoundError:
                sarcs_data = self.sg_tools.time_series.sarc_info_gpr(
                    save_data=False
                )

            sarcs_length = sarcs_data[2]
            sarcs_length_norm = self.sg_tools.time_series.normalize(
                sarcs_data[2]
            )

            signal_th = 0
            signal_dist = 10
            signal_width = 5

            num_sarcs = sarcs_length.shape[0]
            num_frames = sarcs_length.shape[1]
            frames = np.arange(0, num_frames, 1)
            sarc_ids = np.arange(0, num_sarcs, 1)
            pix_length_median = np.zeros(num_sarcs)
            pix_length_mean = np.zeros(num_sarcs)
            pix_length_min = np.zeros(num_sarcs)
            pix_length_max = np.zeros(num_sarcs)
            pix_percent_shortening = np.zeros(num_sarcs)
            mean_contraction_time = np.zeros(num_sarcs)
            mean_relax_time = np.zeros(num_sarcs)
            mean_flat_time = np.zeros(num_sarcs)
            mean_period_len = np.zeros(num_sarcs)
            frames_to_first_peak = np.zeros(num_sarcs)
            peaks_count = np.zeros(num_sarcs)
            for sarc_id, (sarc_length, sarc_length_norm) in enumerate(
                zip(sarcs_length, sarcs_length_norm)
            ):
                sarc_length_med = signal.medfilt(sarc_length_norm, 5)
                sarc_length_deriv = np.gradient(sarc_length_norm, frames)

                thresh_flat = 0.025 * (
                    np.max(sarc_length_med) - np.min(sarc_length_med)
                )
                count_R = np.sum(sarc_length_deriv > thresh_flat)
                count_C = np.sum(sarc_length_deriv < -thresh_flat)
                count_F = num_frames - count_R - count_C

                # detect valleys
                peaks_L, _ = signal.find_peaks(
                    -sarc_length_med,
                    threshold=signal_th,
                    distance=signal_dist,
                    width=signal_width,
                )

                num_peaks = np.sum(
                    sarc_length_med[peaks_L]
                    < np.mean(sarc_length_med) - thresh_flat
                )
                if num_peaks == 0:
                    num_peaks = np.inf

                mean_contraction_time[sarc_id] = count_C / num_peaks
                mean_relax_time[sarc_id] = count_R / num_peaks
                mean_flat_time[sarc_id] = count_F / num_peaks

                min_sarc_length = np.min(sarc_length)
                max_sarc_length = np.max(sarc_length)
                pix_length_median[sarc_id] = np.median(sarc_length)
                pix_length_mean[sarc_id] = np.mean(sarc_length)
                pix_length_min[sarc_id] = min_sarc_length
                pix_length_max[sarc_id] = max_sarc_length
                pix_percent_shortening[sarc_id] = (
                    100 * (max_sarc_length - min_sarc_length) / max_sarc_length
                )

                if peaks_L.size:
                    frames_to_first_peak[sarc_id] = peaks_L[0]
                else:
                    frames_to_first_peak[sarc_id] = 0

                mean_period_len[sarc_id] = num_frames / num_peaks
                peaks_count[sarc_id] = num_peaks

            # import data to a dataframe
            data = np.vstack(
                (
                    sarc_ids,
                    pix_length_median,
                    pix_length_mean,
                    pix_length_min,
                    pix_length_max,
                    pix_percent_shortening,
                    mean_contraction_time,
                    mean_relax_time,
                    mean_flat_time,
                    mean_period_len,
                    frames_to_first_peak,
                    peaks_count,
                )
            )

            cols = [
                "sarc_ids",
                "pix_length_median",
                "pix_length_mean",
                "pix_length_min",
                "pix_length_max",
                "pix_percent_shortening",
                "mean_contraction_time",
                "mean_relax_time",
                "mean_flat_time",
                "mean_period_len",
                "frames_to_first_peak",
                "peaks_count",
            ]

            df = pd.DataFrame(data.T, columns=cols)

            if save_data:
                df.to_pickle(
                    f"./{self.sg_tools.output_dir}/timeseries_params.pkl"
                )

            return df
