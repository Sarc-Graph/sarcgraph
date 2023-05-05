import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import pickle
import imageio
import shutil
import json
import os

from matplotlib.lines import Line2D
from scipy import signal
from scipy.signal import find_peaks
from scipy.linalg import polar
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from pathlib import Path
from typing import Tuple  # List, Union

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

from sarcgraph.sg import SarcGraph

simplefilter("ignore", category=ConvergenceWarning)


class SarcGraphTools:
    def __init__(
        self,
        input_dir: str = "test-run",
        quality: int = 300,
        include_eps: bool = False,
        save_results: bool = True,
    ):
        """Tools for post processing analysis on detected sarcomeres.

        Parameters
        ----------
        input_dir : str
            Should be the same as the `output_dir` in sarcgraph.sg.Sarcgraph(),
             by default "test-run"
        quality : int
            dpi of saved figures in png format, by default 300
        include_eps : bool
            save eps format of figures, by default False
        save_results : bool
            save the results of post processing analysis, by default True
        """
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"{input_dir}/ directory was not found!")

        self.input_dir = input_dir
        self.output_dir = input_dir
        self.quality = quality
        self.include_eps = include_eps
        self.save_results = save_results
        self.visualization = self.Visualization(self)
        self.time_series = self.TimeSeries(self)
        self.analysis = self.Analysis(self)

    ###########################################################
    #                    Time Series Class                    #
    ###########################################################
    class TimeSeries:
        def __init__(self, sg_tools):
            """Provides the tools to apply Gaussian Process Regression (GPR) on
            timeseries of detected sarcomere
            """
            self.sg_tools = sg_tools

        def _dtw_distance(self, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
            """Compute distance based on dynamic time warping (DTW) between
            two 1D signals s1 and s2.

            Parameters
            ----------
            s1 : np.ndarray
                1D signal
            s2 : np.ndarray
                1D signal

            Returns
            -------
            np.ndarray
                DTW distance between s1 and s2 based on euclidean distance.
            """
            if (not isinstance(s1, np.ndarray)) or (
                not isinstance(s2, np.ndarray)
            ):
                raise TypeError("s1 and s2 must be 1D numpy arrays.")
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
            else:
                raise ValueError("s1 and s2 must be 1D numpy arrays.")

        def _gpr(self, s: np.ndarray) -> np.ndarray:
            """Applies Gaussian Process Regression (GPR) on a 1D signal

            Parameters
            ----------
            s : np.ndarray

            Returns
            -------
            np.ndarray

            Notes
            -----
            For more information on GPR check:
            https://scikit-learn.org/stable/modules/gaussian_process.html
            """
            num_frames = len(s)

            kernel = 1.0 * RBF(
                length_scale=1.0, length_scale_bounds=(1e-5, 1e1)
            ) + 1.0 * WhiteKernel(
                noise_level=1.0, noise_level_bounds=(1e-5, 1e1)
            )
            model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
            xdata = np.arange(num_frames).reshape(-1, 1)
            xhat = xdata
            ydata = s.reshape(-1, 1)

            remove_indices = np.where(np.isnan(ydata.reshape(-1)))[0]
            xdata = np.delete(xdata, remove_indices, axis=0)
            ydata = np.delete(ydata, remove_indices, axis=0)

            model.fit(xdata, ydata)
            yhat = model.predict(xhat)

            return yhat.reshape(-1)

        def _sarcomeres_length_normalize(
            self, sarcomeres: pd.DataFrame
        ) -> pd.DataFrame:
            groupby_sarc_id = sarcomeres.groupby("sarc_id")
            mean_length = groupby_sarc_id.length.transform("mean")
            sarcomeres["length_norm"] = (
                sarcomeres.length - mean_length
            ) / mean_length
            return sarcomeres

        def sarcomeres_gpr(self) -> pd.DataFrame:
            """Applies Gaussian Process Regression (GPR) on the output of the
            sarcomere detection algorithm to reduce the noise and fill in
            missing data.

            Returns
            -------
            pd.DataFrame

            Notes
            -----
            For more information on GPR check
            https://scikit-learn.org/stable/modules/gaussian_process.html

            See Also
            --------
            :func:`sarcgraph.sg.SarcGraph.sarcomere_detection`
            """
            sarcomeres = self.sg_tools._load_sarcomeres()
            cols = [
                "x",
                "y",
                "length",
                "angle",
                "width",
            ]

            num_sarcs = sarcomeres.sarc_id.max() + 1
            for info_type in cols:
                for sarc_num in range(num_sarcs):
                    row_mask = sarcomeres.sarc_id == sarc_num
                    s = sarcomeres.loc[row_mask, info_type].to_numpy()
                    sarcomeres.loc[row_mask, info_type] = self._gpr(s)
            sarcomeres = self._sarcomeres_length_normalize(sarcomeres)
            if self.sg_tools.save_results:
                sarcomeres.to_csv(
                    f"./{self.sg_tools.output_dir}/sarcomeres_gpr.csv"
                )
            return sarcomeres

    #############################################################
    #                    Visualization Class                    #
    #############################################################
    class Visualization:
        def __init__(self, sg_tools):
            """Provides tools to visualize the results of post-processing
            analysis on detected sarcomeres.
            """
            self.sg_tools = sg_tools

        def zdiscs_and_sarcs(self, frame_num: int = 0):
            """Visualize and save the plot of segmented zdiscs and detected
            sarcomeres in the chosen frame

            Parameters
            ----------
            frame_num : int, by default 0
            """
            raw_frame = self.sg_tools._load_raw_frames()[frame_num]
            contours = self.sg_tools._load_contours()[frame_num]
            sarcomeres = self.sg_tools._load_sarcomeres_gpr()

            sarcs_x = sarcomeres[sarcomeres.frame == frame_num].x
            sarcs_y = sarcomeres[sarcomeres.frame == frame_num].y

            ax = plt.axes()
            ax.set_aspect("equal")
            ax.imshow(raw_frame[:, :, 0], cmap=plt.cm.gray)
            ax.set_title(
                f"{len(contours)} z-disks and {len(sarcs_x)} sarcomeres "
                f"in frame {frame_num+1}"
            )
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], "#0000cc", linewidth=1)
            ax.plot(sarcs_y, sarcs_x, "*", color="#cc0000", markersize=3)
            ax.set_xticks([])
            ax.set_yticks([])

            # Create legend
            legend_elements = [
                Line2D([0], [0], color="#0000cc", lw=4, label="Z-disc"),
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="#cc0000",
                    markersize=8,
                    linestyle="None",
                    label="Sarcomere",
                ),
            ]

            ax.legend(handles=legend_elements)

            output_file = (
                f"{self.sg_tools.output_dir}/zdiscs-sarcs-frame-{frame_num}"
            )
            plt.savefig(
                f"{output_file}.png",
                dpi=self.sg_tools.quality,
                bbox_inches="tight",
            )

            if self.sg_tools.include_eps:
                plt.savefig(f"{output_file}.eps", bbox_inches="tight")

            plt.show()

        def contraction(self):
            """Visualize all detected sarcomeres in every frame according to
            normalized fraction length and save as a gif file.
            """
            raw_frames = self.sg_tools._load_raw_frames()
            sarcomeres = self.sg_tools._load_sarcomeres_gpr()

            num_frames = sarcomeres.frame.max() + 1
            frames = sarcomeres.frame.to_numpy()
            sarcs_x = sarcomeres.x.to_numpy()
            sarcs_y = sarcomeres.y.to_numpy()
            sarcs_length_norm = sarcomeres.length_norm.to_numpy()

            img_list = []
            Path("tmp").mkdir(parents=True, exist_ok=True)
            for frame_num in range(num_frames):
                indices = frames == frame_num
                raw_frame = raw_frames[frame_num, :, :, 0]

                plt.figure()
                plt.imshow(raw_frame, cmap=plt.cm.gray)
                y = sarcs_x[indices]
                x = sarcs_y[indices]
                length = np.abs(sarcs_length_norm[indices])
                colors = np.piecewise(
                    length,
                    [length < 0.2, length >= 0.2],
                    [lambda x: 2.5 * x + 0.5, lambda x: x],
                )
                for p_x, p_y, p_col in zip(x, y, colors):
                    col = (1 - p_col, 0, p_col)
                    plt.scatter(p_x, p_y, s=15, color=col, marker="o")

                ax = plt.gca()
                ax.set_xticks([])
                ax.set_yticks([])
                plt.savefig(
                    f"tmp/frame-{frame_num}.png",
                    dpi=self.sg_tools.quality,
                    bbox_inches="tight",
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

        def normalized_sarcs_length(self):
            """Plot normalized length of all detected sarcomeres vs frame
            number.
            """
            sarcomeres = self.sg_tools._load_sarcomeres_gpr()

            num_sarcs = sarcomeres.sarc_id.max() + 1
            sarc_ids = sarcomeres.sarc_id.to_numpy()
            sarcs_length_norm = sarcomeres.length_norm.to_numpy()
            groupby_frame = sarcomeres.groupby("frame")
            sarcs_length_norm_median = groupby_frame.length_norm.median()
            sarcs_length_norm_mean = groupby_frame.length_norm.mean()

            _, ax = plt.subplots(figsize=(5, 5))
            ax.grid("on")
            for sarc_id in range(num_sarcs):
                indices = sarc_ids == sarc_id
                plt.plot(
                    sarcs_length_norm[indices],
                    linewidth=0.25,
                    color=(0.545, 0.106, 0.086),
                    alpha=0.1,
                )
            plt.plot(
                sarcs_length_norm_mean,
                "k-",
                linewidth=2,
                label="mean curve",
            )
            plt.plot(
                sarcs_length_norm_median,
                "--",
                color=(0.5, 0.5, 0.5),
                linewidth=2,
                label="median curve",
            )
            plt.xlabel("frame")
            plt.ylabel("normalized length")
            plt.title(
                f"timeseries data, tracked and normalized, {num_sarcs} "
                "sarcomeres"
            )
            plt.ylim((-0.1, 0.1))
            plt.legend()
            plt.tight_layout()
            output_file = (
                f"{self.sg_tools.output_dir}/normalized_sarcomeres_length"
            )
            plt.savefig(
                f"{output_file}.png",
                dpi=self.sg_tools.quality,
                bbox_inches="tight",
            )
            if self.sg_tools.include_eps:
                plt.savefig(f"{output_file}.eps", bbox_inches="tight")

            plt.show()

        def OOP(self):
            """Plot recovered Orientational Order Parameter."""
            OOP = self.sg_tools._load_recovered_info("OOP")

            plt.figure(figsize=(5, 5))
            plt.subplot(1, 1, 1)
            plt.plot(OOP, "k-", label="OOP recovered")
            plt.legend()
            plt.title("recovered Orientational Order Parameter")
            plt.xlabel("frames")

            output_file = f"{self.sg_tools.output_dir}/recovered_OOP"
            plt.savefig(
                f"{output_file}.png",
                dpi=self.sg_tools.quality,
                bbox_inches="tight",
            )
            if self.sg_tools.include_eps:
                plt.savefig(f"{output_file}.eps", bbox_inches="tight")

            plt.show()

        def F(self):
            """Plot recovered deformation gradient."""
            F = self.sg_tools._load_recovered_info("F")

            _, ax = plt.subplots(figsize=(5, 5))
            ax.grid("on")
            plt.plot(
                F[:, 0, 0] - 1,
                "--",
                color=(0.078, 0.118, 0.594),
                linewidth=5,
                label="F11 recovered",
            )
            plt.plot(
                F[:, 1, 1] - 1,
                "--",
                color=(0.545, 0.106, 0.086),
                linewidth=4,
                label="F22 recovered",
            )
            plt.plot(
                F[:, 0, 1],
                ":",
                color=(0.078, 0.118, 0.594),
                label="F12 recovered",
            )
            plt.plot(
                F[:, 1, 0],
                ":",
                color=(0.545, 0.106, 0.086),
                label="F21 recovered",
            )
            plt.legend()
            plt.title("recovered deformation gradient")
            plt.xlabel("frames")
            plt.ylabel("value")
            output_file = f"{self.sg_tools.output_dir}/recovered_F"
            plt.savefig(
                f"{output_file}.png",
                dpi=self.sg_tools.quality,
                bbox_inches="tight",
            )
            if self.sg_tools.include_eps:
                plt.savefig(f"{output_file}.eps", bbox_inches="tight")

            plt.show()

        def J(self):
            """Plot recovered deformation jacobian."""
            J = self.sg_tools._load_recovered_info("J")
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
            # peaks_U, _ = find_peaks(data_med, threshold=th, distance=di,
            # width=wi)
            peaks_L, _ = find_peaks(
                -1.0 * J_med, threshold=0.0, distance=10, width=5
            )
            plt.grid()
            # plt.plot(x[peaks_U],data[peaks_U],'rx',markersize=10)
            plt.plot(frames[peaks_L], J[peaks_L], "rx", markersize=13)
            plt.title(
                f"frames contract: {count_C}, relax: {count_R}, flat: "
                f"{count_F}"
            )
            plt.xlabel("frame number")
            plt.ylabel("determinate of average F")
            plt.tight_layout()

            output_file = f"{self.sg_tools.output_dir}/recovered_J"
            plt.savefig(
                f"{output_file}.png",
                dpi=self.sg_tools.quality,
                bbox_inches="tight",
            )
            if self.sg_tools.include_eps:
                plt.savefig(f"{output_file}.eps", bbox_inches="tight")

            plt.show()

        def F_eigenval_animation(self, no_anim=False):
            """Visualize the eigenvalues of F over all frames"""
            F_all = self.sg_tools._load_recovered_info("F")
            J_all = self.sg_tools._load_recovered_info("J")

            num_frames = len(J_all)
            frames = np.arange(num_frames)

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

            if no_anim:
                return np.array([lambda_1, lambda_2])

            raw_frames = self.sg_tools._load_raw_frames()[:, :, :, 0]
            OOP_all = self.sg_tools._load_recovered_info("OOP")
            OOP_vec_all = self.sg_tools._load_recovered_info("OOP_vector")

            if self.sg_tools.save_results:
                with open(
                    f"{self.sg_tools.output_dir}/recovered_lambda.npy", "wb"
                ) as file:
                    np.save(file, np.array([lambda_1, lambda_2]))

            img_list = []
            Path("tmp").mkdir(parents=True, exist_ok=True)
            radius = 0.2 * np.min(raw_frames.shape[1:])
            th = np.linspace(0, 2.0 * np.pi, 100)
            v = np.array([radius * np.cos(th), radius * np.sin(th)]).T
            center = np.array(raw_frames.shape[1:]).reshape(-1) / 2
            vec_circ = v + center
            p1_1 = center - radius * vec_1
            p1_2 = center + radius * vec_1
            p2_1 = center - radius * vec_2
            p2_2 = center + radius * vec_2
            for frame_num, raw_img in enumerate(raw_frames):
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
                    dpi=self.sg_tools.quality,
                    bbox_inches="tight",
                )
                plt.close()
                img_list.append(imageio.imread(f"tmp/frame-{frame_num}.png"))
            shutil.rmtree("tmp")

            if num_frames > 1:
                imageio.mimsave(
                    f"{self.sg_tools.output_dir}/F_anim.gif", img_list
                )

        def timeseries_params(self):
            """Visualize time series parameters."""
            ts_params = self.sg_tools._load_ts_params()

            plt.figure(figsize=(7, 7))

            med = np.median(ts_params["mean_contraction_time"])
            plt.subplot(2, 2, 1)
            plt.hist(ts_params["mean_contraction_time"])
            plt.plot([med, med], [0, 10], "r--")
            plt.xlabel("frames")
            plt.title(f"median_contract: {med:.2f}")
            plt.tight_layout()

            med = np.median(ts_params["mean_relax_time"])
            plt.subplot(2, 2, 2)
            plt.hist(ts_params["mean_relax_time"])
            plt.plot([med, med], [0, 10], "r--")
            plt.xlabel("frames")
            plt.title(f"median_relax: {med:.2f}")
            plt.tight_layout()

            med = np.median(ts_params["mean_flat_time"])
            plt.subplot(2, 2, 3)
            plt.hist(ts_params["mean_flat_time"])
            plt.plot([med, med], [0, 10], "r--")
            plt.xlabel("frames")
            plt.title(f"median_flat: {med:.2f}")
            plt.tight_layout()

            med = np.median(ts_params["mean_period_len"])
            plt.subplot(2, 2, 4)
            plt.hist(ts_params["mean_period_len"])
            plt.plot([med, med], [0, 10], "r--")
            plt.xlabel("frames")
            plt.title(f"median_period: {med:.2f}")
            plt.tight_layout()

            out_file = f"{self.sg_tools.output_dir}/histogram_time_constants"
            plt.savefig(
                f"{out_file}.png",
                dpi=self.sg_tools.quality,
                bbox_inches="tight",
            )
            if self.sg_tools.include_eps:
                plt.savefig(f"{out_file}.eps")

            plt.show()

        def dendrogram(self, dist_func: str = "dtw"):
            """Cluster timeseries and plot a dendrogram that shows clusters.

            Parameters
            ----------
            dist_func : str, optional, by default "dtw"
                Choose between "euclidean" or "dtw"
            """
            sarcomeres = self.sg_tools._load_sarcomeres_gpr()

            num_frames = sarcomeres.frame.max() + 1
            num_sarcs = sarcomeres.sarc_id.max() + 1
            length = sarcomeres.length_norm.to_numpy().reshape(-1, num_frames)

            if dist_func == "dtw":
                dtw_dist = self.sg_tools.time_series._dtw_distance
                dist_mat = np.zeros((num_sarcs, num_sarcs))
                for sarc_1_id in range(num_sarcs):
                    for sarc_2_id in range(sarc_1_id + 1, num_sarcs):
                        dist = dtw_dist(
                            length[sarc_1_id, :],
                            length[sarc_2_id, :],
                        )
                        dist_mat[sarc_1_id, sarc_2_id] = dist
                        dist_mat[sarc_2_id, sarc_1_id] = dist
            if dist_func == "euclidean":
                dist_mat = squareform(pdist(length, "euclidean"))

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
                plt.plot(length[ix, :] + kk * 0.3, c=col)

            plt.tight_layout()
            plt.ylim((-0.4, kk * 0.3 + 0.35))
            plt.axis("off")

            output_file = f"{self.sg_tools.output_dir}/dendrogram_{dist_func}"
            plt.savefig(f"{output_file}.pdf")

            plt.show()

        def spatial_graph(self):
            """Visualizes the spatial graph

            See Also
            --------
            :func:`sarcgraph.sg_tools.SarcGraphTools.Analysis.create_spatial_graph`
            """
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
            plt.axis("equal")
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

            output_file = f"./{self.sg_tools.output_dir}/spatial-graph"
            plt.savefig(
                f"./{output_file}.png",
                dpi=self.sg_tools.quality,
                bbox_inches="tight",
            )
            if self.sg_tools.include_eps:
                plt.savefig(f"./{output_file}.eps")

            plt.show()

        def tracked_vs_untracked(
            self,
            file_path: str,
            start_frame: int = 0,
            stop_frame: int = np.inf,
        ):
            """Visualize metrics to compare the effect of tracking sarcomeres
            in a video vs only detecting sarcomeres in each frame without
            tracking

            Parameters
            ----------
            file_path: str
                address to the original video file
            start_frame : int, optional, by default 0
            stop_frame : int, optional, by default np.inf
            """
            # process the whole video and detect and track sarcomeres
            sarcomeres = self.sg_tools._load_sarcomeres_gpr()

            total_frames = sarcomeres.frame.max() + 1
            stop_frame = min(stop_frame, total_frames)
            start_frame = min(start_frame, stop_frame - 1)
            num_frames = stop_frame - start_frame

            sarcomeres = sarcomeres[
                sarcomeres.frame.between(start_frame, stop_frame)
            ]
            num_tracked = sarcomeres.sarc_id.max() + 1

            # process the video frame by frame - no tracking
            sg_video = SarcGraph(self.sg_tools.input_dir)
            segmented_zdiscs = sg_video.zdisc_segmentation(file_path)

            length_all_frames = []
            width_all_frames = []
            angle_all_frames = []
            median_length_all_frames = []
            sarc_num_all_frames = []
            sg_image = SarcGraph(self.sg_tools.input_dir, "image")
            for frame in range(start_frame, stop_frame):
                segmented_zdiscs_frame = segmented_zdiscs.loc[
                    segmented_zdiscs.frame == frame
                ].copy()
                segmented_zdiscs_frame.loc[:, "frame"] = 0.0
                tracked_zdiscs_frame = sg_image.zdisc_tracking(
                    segmented_zdiscs=segmented_zdiscs_frame
                )
                sarcomeres_frame, _ = sg_image.sarcomere_detection(
                    tracked_zdiscs=tracked_zdiscs_frame
                )
                length_all_frames.append(sarcomeres_frame.length.to_numpy())
                width_all_frames.append(sarcomeres_frame.width.to_numpy())
                angle_all_frames.append(sarcomeres_frame.angle.to_numpy())
                median_length_all_frames.append(
                    np.median(length_all_frames[-1])
                )
                sarc_num_all_frames.append(sarcomeres_frame.sarc_id.max() + 1)

            # compute average number of not tracked sarcomeres in each frame
            num_not_tracked = np.mean(sarc_num_all_frames)

            len_diff_mean = []
            sarcs_length_grouped = sarcomeres.groupby("frame").length
            for untracked_len, tracked_len in zip(
                length_all_frames, sarcs_length_grouped
            ):
                tracked_len = tracked_len[1].to_numpy()
                tracked_len_mean = np.mean(tracked_len)
                len_diff_mean.append(
                    self.sg_tools.analysis._sampler(
                        untracked_len,
                        tracked_len_mean,
                        num_tracked,
                        num_run=1000,
                    )
                )

            plt.figure(figsize=(np.clip(int(num_frames * 0.3), 10, 25), 5))
            plt.boxplot(
                len_diff_mean, positions=range(start_frame, stop_frame)
            )
            plt.plot([start_frame, stop_frame - 1], [-0.5, -0.5], "k--")
            plt.plot([start_frame, stop_frame - 1], [0.5, 0.5], "k--")
            plt.title(
                f"Comparison of length in pixels, approx {num_not_tracked:.2f}"
                f" untracked, {num_tracked} tracked"
            )
            plt.xlabel("frame number")
            plt.ylabel(r"$\mu_{track}-\mu_{all}$")
            output_file = f"{self.sg_tools.output_dir}/length-comparison"
            plt.savefig(
                f"{output_file}.png",
                dpi=self.sg_tools.quality,
                bbox_inches="tight",
            )
            if self.sg_tools.include_eps:
                plt.savefig(f"{output_file}.eps")

            wid_diff_mean = []
            sarcs_width_grouped = sarcomeres.groupby("frame").width
            for untracked_wid, tracked_wid in zip(
                width_all_frames, sarcs_width_grouped
            ):
                tracked_wid = tracked_wid[1].to_numpy()
                tracked_wid_mean = np.mean(tracked_wid)
                wid_diff_mean.append(
                    self.sg_tools.analysis._sampler(
                        untracked_wid,
                        tracked_wid_mean,
                        num_tracked,
                        num_run=1000,
                    )
                )

            plt.figure(figsize=(np.clip(int(num_frames * 0.3), 10, 25), 5))
            plt.boxplot(
                wid_diff_mean, positions=range(start_frame, stop_frame)
            )
            plt.plot([start_frame, stop_frame - 1], [-0.5, -0.5], "k--")
            plt.plot([start_frame, stop_frame - 1], [0.5, 0.5], "k--")
            plt.title(
                f"Comparison of Width in pixels, approx {num_not_tracked:.2f} "
                f"untracked, {num_tracked} tracked"
            )
            plt.xlabel("frame number")
            plt.ylabel(r"$\mu_{track}-\mu_{all}$")
            output_file = f"{self.sg_tools.output_dir}/width-comparison"
            plt.savefig(
                f"{output_file}.png",
                dpi=self.sg_tools.quality,
                bbox_inches="tight",
            )
            if self.sg_tools.include_eps:
                plt.savefig(f"{output_file}.eps")

            ang_diff_mean = []
            rad_diff_mean = []
            sarcs_angle_grouped = sarcomeres.groupby("frame").angle
            for untracked_ang, tracked_ang in zip(
                angle_all_frames, sarcs_angle_grouped
            ):
                tracked_ang = tracked_ang[1].to_numpy()
                out = self.sg_tools.analysis._angular_mean(tracked_ang)
                tracked_ang_mean = out[0]
                tracked_rad_mean = out[1]
                ang_diff, rad_diff = self.sg_tools.analysis._angular_sampler(
                    untracked_ang,
                    tracked_ang_mean,
                    tracked_rad_mean,
                    num_tracked,
                    num_run=1000,
                )
                ang_diff_mean.append(ang_diff)
                rad_diff_mean.append(rad_diff)

            plt.figure(figsize=(np.clip(int(num_frames * 0.3), 10, 25), 10))
            plt.subplot(2, 1, 1)
            plt.boxplot(
                ang_diff_mean, positions=range(start_frame, stop_frame)
            )
            plt.plot(
                [start_frame, stop_frame - 1], [-np.pi / 8, -np.pi / 8], "k--"
            )
            plt.plot(
                [start_frame, stop_frame - 1], [np.pi / 8, np.pi / 8], "k--"
            )
            plt.title(
                f"Comparison of angle in radians, approx {num_not_tracked:.2f}"
                f" untracked, {num_tracked} tracked"
            )
            plt.xlabel("frame number")
            plt.ylabel(r"$\mu_{track}-\mu_{all}$")
            plt.subplot(2, 1, 2)
            plt.boxplot(
                rad_diff_mean, positions=range(start_frame, stop_frame)
            )
            plt.plot(
                [start_frame, stop_frame - 1], [0, 0], "r--", label="uniform"
            )
            plt.plot(
                [start_frame, stop_frame - 1], [1, 1], "k--", label="oriented"
            )
            plt.title(
                "Comparison of angle radius in pixels, approx "
                f"{num_not_tracked:.2f} untracked, {num_tracked} tracked"
            )
            plt.xlabel("frame number")
            plt.ylabel(r"$\mu_{track}-\mu_{all}$")
            plt.legend()
            output_file = f"{self.sg_tools.output_dir}/angle-comparison"
            plt.savefig(
                f"{output_file}.png",
                dpi=self.sg_tools.quality,
                bbox_inches="tight",
            )
            if self.sg_tools.include_eps:
                plt.savefig(f"{output_file}.eps")

            plt.show()

    ##########################################################
    #                     Analysis Class                     #
    ##########################################################
    class Analysis:
        def __init__(self, sg_tools):
            """Provides tools for post processing analysis of detected
            sarcomeres.
            """
            self.sg_tools = sg_tools

        def _sampler(
            self,
            signal: np.ndarray,
            mu: float,
            tracked_num: int,
            num_run: int = 1000,
        ) -> np.ndarray:
            """Random sampler

            Parameters
            ----------
            signal : np.ndarray
            mu : float
            tracked_num : int
            num_run : int, optional
            """
            samples = np.zeros(num_run)
            for run in range(num_run):
                ids = np.random.randint(0, len(signal), size=(tracked_num))
                samples[run] = mu - np.mean(signal[ids])
            return samples

        def _angular_mean(self, signal: np.ndarray) -> Tuple[float, float]:
            """Angular signal averaging

            Parameters
            ----------
            signal : np.ndarray

            Returns
            -------
            Tuple[float, float]
                angle and radius of averaged signal
            """
            x_mean = np.nanmean(np.cos(signal))
            y_mean = np.nanmean(np.sin(signal))

            mean_angle = np.arctan2(y_mean, x_mean)
            mean_rad = np.linalg.norm([x_mean, y_mean], 2)

            return mean_angle, mean_rad

        def _angular_sampler(
            self,
            signal: np.ndarray,
            mu_ang: float,
            mu_rad: float,
            tracked_num: int,
            num_run: int = 1000,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Angular random sampler

            Parameters
            ----------
            signal : np.ndarray
            mu_ang : float
            mu_rad : float
            tracked_num : int
            num_run : int, by default 1000, optional

            Returns
            -------
            Tuple[np.ndarray, np.ndarray]
            """
            ang_samples = np.zeros(num_run)
            rad_samples = np.zeros(num_run)
            for run in range(num_run):
                ids = np.random.randint(0, len(signal), size=(tracked_num))
                ang_mean, rad_mean = self.sg_tools.analysis._angular_mean(
                    signal[ids]
                )
                ang_samples[run] = mu_ang - ang_mean
                rad_samples[run] = mu_rad - rad_mean

            return ang_samples, rad_samples

        def compute_F_J(
            self, adjust_reference: bool = False
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Compute the average deformation gradient (F) and its jacobian
            (J) for the whole movie.

            Parameters
            ----------
            adjust_reference : bool, by default False
                The refrence frame by default is the first frame of the movie,
                if this variable is set to ``True`` the function will run twice
                and the refrence frame on the second run will be the most
                contracted frame

            Returns
            -------
            Tuple(np.ndarray, np.ndarray)
                The average deformation gradient (F), shape=(num_frames, 2, 2)
                The jacobian of F, shape=(num_frames)
            """
            sarcomeres = self.sg_tools._load_sarcomeres_gpr()

            sarcs_x = sarcomeres.x.to_numpy()
            sarcs_y = sarcomeres.y.to_numpy()

            # compute Lambda from x_pos and y_pos
            num_frames = sarcomeres.frame.max() + 1
            num_sarcs = sarcomeres.sarc_id.max() + 1
            n = int(num_sarcs * (num_sarcs - 1) / 2)

            Lambda_list = []
            for frame_num in range(num_frames):
                ids = sarcomeres.frame == frame_num
                x_vec = sarcs_x[ids]
                y_vec = sarcs_y[ids]

                Lambda = np.zeros((2, n))
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

            if self.sg_tools.save_results:
                np.save(f"{self.sg_tools.output_dir}/recovered_F.npy", F_all)
                np.save(f"{self.sg_tools.output_dir}/recovered_J.npy", J_all)

            return F_all, J_all

        def compute_OOP(self) -> Tuple[np.ndarray, np.ndarray]:
            """Computes Orientation Order Parameter (OOP) for the whole movie.

            Returns
            -------
            Tuple[np.ndarray, np.ndarray]
                OOP for all frames, shape=(num_frames)
                OOP vector for all frames, shape=(num_frames, 2)
            """

            def f(data):
                rxx = np.cos(data) ** 2
                rxy = np.cos(data) * np.sin(data)
                ryy = np.sin(data) ** 2
                n = np.array([[rxx, rxy], [rxy, ryy]])
                t = 2 * n - np.eye(2).reshape(2, 2, 1)
                return np.mean(t, axis=2)

            sarcomeres = self.sg_tools._load_sarcomeres_gpr()
            num_frames = sarcomeres.frame.max() + 1

            T = sarcomeres.groupby("frame")["angle"].apply(lambda x: f(x))
            OOP_all = np.zeros(num_frames)
            OOP_vec_all = np.zeros((num_frames, 2))
            for frame in range(num_frames):
                u, v = np.linalg.eig(T[frame])
                OOP_all[frame] = np.max(u)
                OOP_vec_all[frame, :] = v[:, np.argmax(u)]

            if self.sg_tools.save_results:
                np.save(
                    f"{self.sg_tools.output_dir}/recovered_OOP.npy", OOP_all
                )
                np.save(
                    f"{self.sg_tools.output_dir}/recovered_OOP_vector.npy",
                    OOP_vec_all,
                )

            return OOP_all, OOP_vec_all

        def compute_metrics(self, frame: int = None) -> dict:
            """This function computes the following metrics as defind in the
            paper: {OOP, C_iso, C_OOP, s_til, s_avg}.

            Parameters
            ----------
            frame : int, optional
                By default is set to the frame with maximum contraction

            Notes
            -----
            See the paper for more information:
            https://arxiv.org/abs/2102.02412

            Returns
            -------
            dict
            """

            def f(data):
                return (np.max(data) - np.min(data)) / (np.max(data) + 1)

            F_all = self.sg_tools._load_recovered_info("F")
            J_all = self.sg_tools._load_recovered_info("J")
            OOP_all = self.sg_tools._load_recovered_info("OOP")
            OOP_vec_all = self.sg_tools._load_recovered_info("OOP_vector")
            sarcomeres = self.sg_tools._load_sarcomeres_gpr()

            if frame is None:
                frame = np.argmin(J_all)
            OOP = OOP_all[frame]
            OOP_vec = OOP_vec_all[frame]
            F = F_all[frame]
            J = J_all[frame]

            v = OOP_vec
            v_abs = np.linalg.norm(v, 2)
            v0 = np.dot(np.linalg.inv(F), v)
            v0_abs = np.linalg.norm(v0, 2)
            avg_contract = 1 - np.sqrt(J)
            avg_aligned_contract = (v0_abs - v_abs) / v0_abs

            sarcs_groupby_sarc_id = sarcomeres.groupby("sarc_id").length_norm
            sarcs_groupby_frame = sarcomeres.groupby("frame").length_norm

            s_til = sarcs_groupby_sarc_id.apply(lambda x: f(x)).median()
            s_avg = f(sarcs_groupby_frame.mean())

            info_dict = {
                "OOP": OOP,
                "C_iso": avg_contract,
                "C_OOP": avg_aligned_contract,
                "s_til": s_til,
                "s_avg": s_avg,
            }

            if self.sg_tools.save_results:
                with open(
                    f"{self.sg_tools.output_dir}/recovered_metrics.json", "w"
                ) as file:
                    json.dump(info_dict, file)

            return info_dict

        def compute_ts_params(self) -> pd.DataFrame:
            """Compute and save timeseries time constants (contraction time,
            relaxation time, flat time, period, offset, etc.).

            Returns
            -------
            pd.DataFrame
            """
            sarcomeres = self.sg_tools._load_sarcomeres_gpr()
            num_sarcs = sarcomeres.sarc_id.max() + 1
            num_frames = sarcomeres.frame.max() + 1

            sarcs_length = sarcomeres.groupby("sarc_id").length
            sarcs_length_norm = sarcomeres.groupby("sarc_id").length_norm

            signal_th = 0
            signal_dist = 10
            signal_width = 5

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
            for length, length_norm in zip(sarcs_length, sarcs_length_norm):
                sarc_id = length[0]
                sarc_length = length[1].to_numpy()
                sarc_length_norm = length_norm[1].to_numpy()

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

            if self.sg_tools.save_results:
                df.to_csv(
                    f"./{self.sg_tools.output_dir}/timeseries_params.csv"
                )

            return df

        def create_spatial_graph(
            self,
            file_path: str = None,
            tracked_zdiscs: pd.DataFrame = None,
        ):
            """Generates and saves a spatial graph of tracked zdiscs where
            edges indicate sarcomeres and edge weights indicates the ratio of
            the frames in which that sarcomere is detected

            Parameters
            ----------
            file_path : str
                The address of an image or a video file to be loaded
            tracked_zdiscs : pd.DataFrame
                Information of all detected and tracked zdiscs in every frame.
            """
            sg_video = SarcGraph(file_type="video")

            # load tracked zdiscs:
            if tracked_zdiscs is None:
                if file_path is None:
                    raise ValueError(
                        "either file_path or "
                        + "tracked_zdiscs should be specified."
                    )
                tracked_zdiscs = sg_video.zdisc_tracking(
                    file_path, save_output=False
                )

            # initiate the graph:
            G = nx.Graph()

            pos = {}
            for particle in tracked_zdiscs.particle.unique():
                x_pos = tracked_zdiscs[tracked_zdiscs.particle == particle][
                    "x"
                ].mean()
                y_pos = tracked_zdiscs[tracked_zdiscs.particle == particle][
                    "y"
                ].mean()
                G.add_node(particle, x_pos=x_pos, y_pos=y_pos)
                pos.update({particle: (y_pos, -x_pos)})

            # SarcGraph object that work with single frames
            sg_image = SarcGraph(file_type="image")

            # frame by frame sarcomere detection, add graph edges and weigts:
            for frame in tracked_zdiscs.frame.unique():
                tracked_zdiscs_frame = tracked_zdiscs[
                    tracked_zdiscs.frame == frame
                ]
                tracked_zdiscs_frame.loc[:]["frame"] = 0
                _, myofibrils = sg_image.sarcomere_detection(
                    tracked_zdiscs=tracked_zdiscs_frame, save_output=False
                )
                for myo in myofibrils:
                    for edge in myo.edges:
                        disc_1 = myo.nodes[edge[0]]["particle_id"]
                        disc_2 = myo.nodes[edge[1]]["particle_id"]
                        if G.has_edge(disc_1, disc_2):
                            G[disc_1][disc_2]["weight"] += 1
                        else:
                            G.add_edge(disc_1, disc_2, weight=1)

            # graph pruning based on minimum weight threshold
            num_frames = tracked_zdiscs.frame.max() + 1
            weight_cutoff = np.floor(0.1 * num_frames)
            edges, weights = zip(*nx.get_edge_attributes(G, "weight").items())
            for edge in edges:
                if G[edge[0]][edge[1]]["weight"] < weight_cutoff:
                    G.remove_edge(edge[0], edge[1])

            # isolated nodes removal
            isolated_nodes = list(nx.isolates(G))
            G.remove_nodes_from(isolated_nodes)

            # save the graph
            output_file = f"{self.sg_tools.output_dir}/spatial-graph"
            nx.write_gpickle(G, path=f"{output_file}.pkl")
            with open(f"{output_file}-pos.pkl", "wb") as file:
                pickle.dump(pos, file)

    ###########################################################
    #                    Utility Functions                    #
    ###########################################################
    def _run_all(self, file_path: str = None):
        """Runs all functions in the Analysis class and saves outputs.

        Parameters
        ----------
        file_path : str
            Path to the input video or image file.
        """
        self.analysis.compute_F_J()
        self.analysis.compute_OOP()
        self.analysis.compute_metrics()
        self.analysis.compute_ts_params()
        self.analysis.create_spatial_graph(file_path)

    def _load_raw_frames(self):
        try:
            raw_frames = np.load(f"{self.input_dir}/raw-frames.npy")
        except Exception:
            self._raise_sarcgraph_data_not_found("raw-frames.npy")
        return raw_frames

    def _load_contours(self):
        try:
            contours = np.load(
                f"{self.input_dir}/contours.npy",
                allow_pickle=True,
            )
        except Exception:
            self._raise_sarcgraph_data_not_found("contours.npy")
        return contours

    def _load_sarcomeres(self):
        try:
            sarcomeres = pd.read_csv(
                f"{self.input_dir}/sarcomeres.csv", index_col=[0]
            )
        except Exception:
            self._raise_sarcgraph_data_not_found("sarcomeres.csv")
        return sarcomeres

    def _load_sarcomeres_gpr(self):
        try:
            sarcomeres = pd.read_csv(
                f"{self.input_dir}/sarcomeres_gpr.csv",
                index_col=[0],
            )
        except Exception:
            self._raise_data_not_found("sarcomeres_gpr.csv")
        return sarcomeres

    def _load_recovered_info(self, info_type: str):
        try:
            OOP = np.load(f"{self.input_dir}/recovered_{info_type}.npy")
        except Exception:
            self._raise_data_not_found(f"recovered_{info_type}.npy")
        return OOP

    def _load_ts_params(self):
        try:
            ts_params = pd.read_csv(
                f"{self.input_dir}/timeseries_params.csv",
                index_col=[0],
            )
        except FileNotFoundError:
            self._raise_data_not_found("timeseries_params.csv")
        return ts_params

    def _raise_sarcgraph_data_not_found(self, data_file: str):
        raise FileNotFoundError(
            f"{data_file} was not found in {self.input_dir}/. Run "
            "SarcGraph().sarcomeres_detection(save_output=True) first."
        )

    def _raise_data_not_found(self, data_file: str):
        raise FileNotFoundError(
            f"{data_file} was not found in {self.input_dir}/. Run "
            "SarcGraphTools()._run_all() first."
        )
