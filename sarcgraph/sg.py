import numpy as np
import pandas as pd
import trackpy as tp
import networkx as nx

import skvideo.io
import skimage.io
import skvideo.utils

from skimage.filters import laplace, gaussian, threshold_otsu
from skimage import measure
from sklearn.cluster import OPTICS
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix
from pathlib import Path
from typing import List, Union, Tuple


class SarcGraph:
    def __init__(self, output_dir: str = "test-run", file_type: str = "video"):
        """Zdiscs and sarcomeres segmentation and tracking.

        Attributes
        ----------
        output_dir : str, optional
            location to save processed information, by default ``'test-run'``
        file_type : str, optional
            use ``'image'`` for single-frame samples and ``'video'`` for
            multi-frame samples, by default ``'video'``
        """
        if file_type not in ["video", "image"]:
            raise ValueError(
                f"{file_type} is not recognized as a valid file_type. Choose "
                "from ['video', 'image']."
            )
        if not isinstance(output_dir, str):
            raise TypeError("output_dir must be a string.")
        Path(f"{output_dir}").mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.file_type = file_type

    ###########################################################
    #                    Utility Functions                    #
    ###########################################################
    def _data_loader(self, file_path: str) -> np.ndarray:
        """Loads a video/image file.

        Parameters
        ----------
        file_path : str
            A video/image file address

        Returns
        -------
        numpy.ndarray
            All frames in the raw format
        """
        if self.file_type == "video":
            data = skvideo.io.vread(file_path)
            if data.shape[0] > 1:
                return data
        return skimage.io.imread(file_path)

    def _to_gray(self, frames: np.ndarray) -> np.ndarray:
        """Convert RGB frames to grayscale.

        Parameters
        ----------
        frames : np.ndarray
            RGB frames

        Returns
        -------
        numpy.ndarray
            All frames in gray scale
        """
        frames_gray = skvideo.utils.rgb2gray(frames)
        if self.file_type == "video" and frames_gray.shape[0] < 2:
            raise ValueError(
                "Failed to load video correctly! Manually load the video into "
                "a numpy array and input to the function as raw_frames."
            )
        if self.file_type == "image" and frames_gray.shape[0] > 1:
            raise ValueError(
                "Trying to load a video while file_type='image'. Load the "
                "image manually or change the file_type to 'video."
            )

        return frames_gray

    def _save_numpy(
        self, data: Union[List, np.ndarray], file_name: str
    ) -> None:
        """Saves a numpy array.

        Parameters
        ----------
        data : np.ndarray
        file_name: str
        """
        if not isinstance(file_name, str):
            raise TypeError("file_name must be a string.")
        if isinstance(data, np.ndarray) or isinstance(data, List):
            np.save(
                f"./{self.output_dir}/{file_name}.npy", data, allow_pickle=True
            )
            return

        raise TypeError("data must be a numpy.ndarray or a List.")

    def _save_dataframe(self, data: pd.DataFrame, file_name: str) -> None:
        """Saves a pandas dataframe.

        Parameters
        ----------
        data : pd.DataFrame
        file_name: str
        """
        if not isinstance(file_name, str):
            raise TypeError("file_name must be a string.")
        if isinstance(data, pd.DataFrame):
            data.to_csv(f"./{self.output_dir}/{file_name}.csv")
            return
        raise TypeError(
            f"'data' type is {type(data)}. 'data' must be a pandas DataFrame."
        )

    def _filter_frames(
        self, frames: np.ndarray, sigma: float = 1.0
    ) -> np.ndarray:
        """Convolves all frames with laplacian and gaussian filters.

        Parameters
        ----------
        frames : np.ndarray
        sigma : float
            Standard deviation for Gaussian kernel

        Returns
        -------
        np.ndarray, shape=(frames, dim_1, dim_2)
        """
        if frames.ndim > 4 or frames.ndim < 3:
            raise ValueError(
                "Input array must have the shape (frames, dim_1, dim_2, "
                "channels=1 (optional))"
            )
        if frames.ndim == 4:
            if frames.shape[3] != 1:
                raise ValueError(
                    "Number of channels in the input array must be 1. Input "
                    "shape: (frames, dim_1, dim_2, channels=1 (optional))"
                )
            frames = frames[:, :, :, 0]
        filtered_frames = np.zeros(frames.shape)
        for i, frame in enumerate(frames):
            laplacian = laplace(frame)
            filtered_frames[i] = gaussian(laplacian, sigma=sigma)
        return filtered_frames

    #############################################################
    #                    Z-Disc Segmentation                    #
    #############################################################
    def _process_input(
        self,
        file_path: Union[str, None] = None,
        raw_frames: Union[np.ndarray, None] = None,
        sigma: float = 1.0,
        save_output: bool = True,
    ) -> np.ndarray:
        """Loads a video/image and filter all frames.

        Parameters
        ----------
        file_path : str
            The address of an image or a video file to be loaded
        raw_frames  : np.ndarray, shape=(frames, dim_1, dim_2, channels)
            Raw input image or video given as a 4 dimensional array
        sigma : float
            Standard deviation for Gaussian kernel
        save_output : bool
            by default True

        Returns
        -------
        np.ndarray, shape=(frames, dim_1, dim_2)
        """
        if raw_frames is None:
            if file_path is None:
                raise ValueError(
                    "Either file_path or raw_frames should be given as"
                    " input."
                )
            else:
                try:
                    raw_frames = self._data_loader(file_path)
                except Exception:
                    raise ValueError(
                        f"Not able to load a file from {file_path}."
                    )

        if not isinstance(raw_frames, np.ndarray):
            raise TypeError("raw_frames must be a numpy array.")
        raw_frames_gray = self._to_gray(raw_frames)
        raw_frames_filtered = self._filter_frames(raw_frames_gray, sigma)
        if save_output:
            self._save_numpy(raw_frames_gray, file_name="raw-frames")
            self._save_numpy(raw_frames_filtered, file_name="filtered-frames")
        return raw_frames_filtered

    def _detect_contours(
        self,
        filtered_frames: np.ndarray,
        min_length: int = 8,
        save_output: bool = True,
    ) -> List[np.ndarray]:
        """Returns contours of detected zdiscs in each frame.

        Parameters
        ----------
        filtered_frames : np.ndarray, shape=(frames, dim_1, dim_2)
        min_length : int
            Zdisc contours shorter than ``min_length`` pixels will be ignored.
        save_output : bool
            by default True

        Returns
        -------
        List[np.ndarray]

        Notes
        -----
        .. warning::
            Here we use the ``skimage.measure.find_contours`` function to find
            contours. This function returns contours with coordinates in
            ``(row, column)`` order, which corresponds to ``(y, x)`` in
            Cartesian coordinates. Therefore, in the rest of the code, we use
            ``y`` for the first dimension and ``x`` for the second dimension.
            For example, for plotting we use ``plt.plot(y, x)``.
        """
        if filtered_frames.ndim != 3:
            raise ValueError(
                "The input must be a 3d numpy array: (frames, " "dim 1, dim 2)"
            )
        if not isinstance(min_length, int):
            try:
                min_length = int(min_length)
            except ValueError:
                raise ValueError("min_length must be an integer.") from None
        length_checker = np.vectorize(len)
        valid_contours = []
        for frame in filtered_frames:
            contour_thresh = threshold_otsu(frame)
            contours = np.array(
                measure.find_contours(frame, contour_thresh), dtype="object"
            )
            contours_size = list(length_checker(contours))
            valid_contours.append(
                contours[np.greater_equal(contours_size, min_length)]
            )
        valid_contours = np.array(valid_contours, dtype="object")
        if save_output:
            self._save_numpy(valid_contours, file_name="contours")
        return valid_contours

    def _process_contour(self, contour: Union[List, np.ndarray]) -> np.ndarray:
        """Computes the center location of a zdisc as well as its two end
        points given its 2d contour.

        Parameters
        ----------
        contour : np.ndarray, shape=(contour_length, 2)

        Returns
        -------
        np.ndarray, shape=(6,)
            zdisc center, end point 1, end point 2
        """
        if len(contour) < 3:
            raise ValueError("A contour must have at least 3 coordinates.")
        # center of a contour
        center_coords = np.mean(contour, axis=0)
        # coordinates of the two points on the contour with maximum distance
        dist_mat = distance_matrix(contour, contour)
        indices = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
        p1, p2 = contour[indices[0]], contour[indices[1]]
        return np.hstack((center_coords, p1, p2))

    def _zdiscs_to_pandas(self, zdiscs_all: List[np.ndarray]) -> pd.DataFrame:
        """Creates a pandas dataframe from the information of detected zdiscs
        in all frames.

        Parameters
        ----------
        zdiscs_all : List[np.ndarray]
            A list of numpy arrays each containing information of detected
            zdiscs in a frame, (zdisc center, end point 1, end point 2)

        Returns
        -------
        pd.DataFrame
            Columns are ``frame`` (frame number), ``x`` and ``y`` (zdiscs
            center position), ``p1_x``, ``p1_y``, ``p2_x``, ``p2_y`` (zdiscs
            end points positions).
        """
        if self.file_type == "video" and len(zdiscs_all) < 2:
            raise ValueError(
                "Only one frame detected. Video is not loaded correctly."
            )
        if self.file_type == "image" and len(zdiscs_all) > 1:
            raise ValueError(
                "More than one frame detected. The image is not loaded "
                "correctly."
            )
        data_frame = []
        for i, zdiscs_frame in enumerate(zdiscs_all):
            if type(zdiscs_frame) != np.ndarray:
                raise TypeError("Input should be a list of numpy arrays.")
            if zdiscs_frame.ndim != 2 or zdiscs_frame.shape[1] != 6:
                raise ValueError(
                    "Each numpy array must have the shape: (number of zdiscs, "
                    "6)"
                )
            frame_id = i * np.ones((len(zdiscs_frame), 1), dtype=int)
            zdiscs_frame_extended = np.hstack((frame_id, zdiscs_frame))
            data_frame.append(
                pd.DataFrame(
                    zdiscs_frame_extended,
                    columns=[
                        "frame",
                        "x",
                        "y",
                        "p1_x",
                        "p1_y",
                        "p2_x",
                        "p2_y",
                    ],
                )
            )
        return pd.concat(data_frame)

    def zdisc_segmentation(
        self,
        file_path: Union[str, None] = None,
        raw_frames: Union[np.ndarray, None] = None,
        sigma: float = 1.0,
        min_length: int = 8,
        save_output: bool = True,
    ) -> pd.DataFrame:
        """Finds Z-Discs in a video/image and extracts related information,
        (frame number, center position, end points).

        Parameters
        ----------
        file_path : str
            address of the input video/image file
        raw_frames  : np.ndarray, shape=(frames, dim_1, dim_2, channels)
            Raw input image or video given as a 4 dimensional array
        sigma : float
            Standard deviation for Gaussian kernel
        min_length : int
            Minimum length for zdisc contours measured in pixels
        save_output : bool
            by default True

        Returns
        -------
        pd.DataFrame
            Information of all detected zdiscs in every frame. Columns are
            ``'frame'`` (frame number), ``'x'`` and ``'y'`` (zdiscs center
            position), ``'p1_x'``, ``'p1_y'``, ``'p2_x'``, ``'p2_y'`` (zdiscs
            end points positions).
        """
        filtered_frames = self._process_input(
            file_path, raw_frames, sigma, save_output
        )
        contours_all = self._detect_contours(
            filtered_frames, min_length, save_output
        )
        zdiscs_all = []
        for contours_frame in contours_all:
            zdiscs_frame = np.zeros((len(contours_frame), 6))
            for i, contour in enumerate(contours_frame):
                zdiscs_frame[i] = self._process_contour(contour)
            zdiscs_all.append(zdiscs_frame)
        zdiscs_all_dataframe = self._zdiscs_to_pandas(zdiscs_all)
        if save_output:
            self._save_dataframe(
                zdiscs_all_dataframe, file_name="segmented-zdiscs"
            )
        return zdiscs_all_dataframe

    #########################################################
    #                    Z-disc Tracking                    #
    #########################################################
    def _merge_tracked_zdiscs(
        self,
        tracked_zdiscs: pd.DataFrame,
        full_track_ratio: float = 0.75,
    ) -> pd.DataFrame:
        """A post processing step to group related partially tracked zdiscs
        using the OPTICS algorithm. Increases the robustness of zdisc tracking
        as well as the number of fully tracked zdiscs.

        Parameters
        ----------
        tracked_zdiscs : pd.DataFrame
            tracked zdiscs information for all frames
        full_track_ratio : float, optional
            If a tracked zdisc appears in enough number of frames defined by
            this ratio, it is considered a fully tracked zdisc. The default
            value 0.75 means if a zdisc is tracked in more than 75 percent of
            all frames it is fully tracked.

        Returns
        -------
        pd.DataFrame

        Notes
        -----
        For a detailed description of the OPTICS algorithm check:
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html
        """
        num_frames = tracked_zdiscs.frame.max() + 1
        tracked_zdiscs_grouped = tracked_zdiscs.groupby("particle")["particle"]
        tracked_zdiscs["freq"] = tracked_zdiscs_grouped.transform("count")
        fully_tracked_zdiscs = tracked_zdiscs.loc[
            tracked_zdiscs.freq == num_frames
        ]
        partially_tracked_zdiscs = tracked_zdiscs.loc[
            tracked_zdiscs.freq < num_frames
        ]

        if partially_tracked_zdiscs.empty:
            return fully_tracked_zdiscs

        partially_tracked_clusters = (
            partially_tracked_zdiscs[["x", "y", "particle"]]
            .groupby(by=["particle"])
            .mean()
        )

        # merge related clusters (neighbors)
        all_clusters_xy = (
            tracked_zdiscs.groupby("particle").mean()[["x", "y"]].to_numpy()
        )
        clusters_min_dist = np.min(
            distance_matrix(all_clusters_xy, all_clusters_xy)
            + 1e6 * np.eye(len(all_clusters_xy)),
            axis=1,
        )
        optics_max_eps = np.mean(clusters_min_dist)
        data = np.array(partially_tracked_clusters)
        optics_model = OPTICS(max_eps=optics_max_eps, min_samples=2)
        optics_result = optics_model.fit_predict(data)
        optics_clusters = np.unique(optics_result)

        all_merged_zdiscs = []
        for i, optics_cluster in enumerate(optics_clusters):
            index = np.where(optics_result == optics_cluster)[0]
            particles_in_cluster = partially_tracked_clusters.iloc[
                index
            ].index.to_numpy()
            if optics_cluster >= 0:
                merged_zdiscs = (
                    partially_tracked_zdiscs.loc[
                        partially_tracked_zdiscs["particle"].isin(
                            particles_in_cluster
                        )
                    ]
                    .groupby("frame")
                    .mean()
                )
                if len(merged_zdiscs) > num_frames * full_track_ratio:
                    merged_zdiscs.particle = -(i + 1)
                    merged_zdiscs.freq = len(merged_zdiscs)
                    all_merged_zdiscs.append(merged_zdiscs.reset_index())
            else:
                for p in particles_in_cluster:
                    no_merge_zdiscs = partially_tracked_zdiscs.loc[
                        partially_tracked_zdiscs["particle"] == p
                    ]
                    if len(no_merge_zdiscs) > num_frames * full_track_ratio:
                        all_merged_zdiscs.append(no_merge_zdiscs)
        if all_merged_zdiscs:
            all_merged_zdiscs = pd.concat(all_merged_zdiscs)
            return pd.concat((fully_tracked_zdiscs, all_merged_zdiscs))
        else:
            return fully_tracked_zdiscs

    def zdisc_tracking(
        self,
        file_path: str = None,
        raw_frames: np.array = None,
        segmented_zdiscs: pd.DataFrame = None,
        sigma: float = 1.0,
        min_length: int = 8,
        tp_depth: float = 4.0,
        full_track_ratio: float = 0.75,
        skip_merging: bool = False,
        save_output: bool = True,
    ) -> pd.DataFrame:
        """Track detected Z-Discs in a video. The input could be the address to
        a video/image sample (``file_path``), raw frames as a numpy array
        (``raw_frames``), or segmented zdiscs information in a pandas datafram
        (``segmented_zdiscs``)e. If the function is run with no inputs, it will
        search for 'raw-frames.npy', or 'segmented-zdiscs.csv' in the
        specified output directory ``SarcGraph().output_dir``.

        Parameters
        ----------
        file_path : str
            The address of an image or a video file to be loaded
        raw_frames : np.ndarray, shape=(frames, dim_1, dim_2, channels)
            Raw input image or video given as a 4 dimensional array
        segmented_zdiscs : pd.DataFrame
            Information of all detected zdiscs in every frame.
        sigma : float
            Standard deviation for Gaussian kernel, by default ``1.0``
        min_length : int
            Minimum length for zdisc contours measured in pixels, by default
            ``8``
        tp_depth : float, optional
            the maximum distance features can move between frames, by default
            ``4.0``
        full_track_ratio : float, optional
            by default ``0.75``
        skip_merging : bool, optional
            skipping the merging step will result in fewer fully tracked
            zdiscs, by default ``False``
        save_output : bool
            by default ``True``

        Returns
        -------
        pd.DataFrame
            tracked zdiscs information. Columns are ``'frame'`` (frame number),
            ``'x'`` and ``'y'`` (zdiscs center position), ``'p1_x'``,
            ``'p1_y'``, ``'p2_x'``, ``'p2_y'`` (zdiscs end points positions),
            and ``'particle'`` (id of each tracked zdisc).

        Notes
        -----
        - If ``SarcGraph().file_type='image'``, tracking will be skipped.

        - For a detailed description of the Trackpy package check:
            http://soft-matter.github.io/trackpy/v0.5.0/tutorial.html

        - For a detailed description of the OPTICS algorithm check:
            https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html

        See Also
        --------
        :func:`sarcgraph.sg.SarcGraph.zdisc_segmentation`

        :func:`sarcgraph.sg.SarcGraph._merge_tracked_zdiscs`

        """
        if segmented_zdiscs is None:
            segmented_zdiscs = self.zdisc_segmentation(
                file_path, raw_frames, sigma, min_length, save_output
            )

        if type(segmented_zdiscs) != pd.DataFrame:
            raise TypeError("segmented_zdiscs shoulb be a dataframe.")

        correct_columns = set(("frame", "x", "y")).issubset(
            set(segmented_zdiscs.columns)
        )

        if not correct_columns:
            raise ValueError(
                "segmented_zdiscs must contain at least three columns: 'x', "
                "'y', 'frame'."
            )

        # skip tracking if the sample is an image
        if self.file_type == "image":
            segmented_zdiscs["particle"] = np.arange(len(segmented_zdiscs))
            tracked_zdiscs = segmented_zdiscs
        else:
            # run tracking with the trackpy package:
            num_frames = len(segmented_zdiscs.frame.unique())
            t = tp.link_df(
                segmented_zdiscs, search_range=tp_depth, memory=num_frames
            )
            tracked_zdiscs = tp.filter_stubs(t, 2).reset_index(drop=True)
            if skip_merging is False:
                tracked_zdiscs = self._merge_tracked_zdiscs(
                    tracked_zdiscs,
                    full_track_ratio,
                )

        if save_output:
            self._save_dataframe(tracked_zdiscs, "tracked-zdiscs")

        return tracked_zdiscs

    #############################################################
    #                    Sarcomere Detection                    #
    #############################################################
    def _zdisc_to_graph(self, zdiscs: np.array, K: int = 3) -> nx.Graph:
        """Creates a graph with zdiscs as nodes. Each zdisc is connected to its
        ``K`` nearest neighbors.

        Parameters
        ----------
        zdiscs : np.array, shape=(N, 3)
            zdiscs information as an array. The first two columns are the x and
            y location of zdisc centers and the last is the particle id.
        K : int, optional
            number of nearest neighbors for each zdisc, by default 3

        Returns
        -------
        nx.Graph
        """
        num_nodes = len(zdiscs)
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        # add the position of each zdisc as a node attribute to the graph
        nodes_pos_dict = {i: pos for i, pos in enumerate(zdiscs[:, 0:2])}
        nodes_particle_dict = {j: id for j, id in enumerate(zdiscs[:, -1])}

        nx.set_node_attributes(G, values=nodes_pos_dict, name="pos")
        nx.set_node_attributes(
            G, values=nodes_particle_dict, name="particle_id"
        )

        # find K nearest zdisc to each zdisc
        neigh = NearestNeighbors(n_neighbors=2)
        neigh.fit(zdiscs[:, 0:2])
        nearestNeighbors = neigh.kneighbors(
            zdiscs[:, 0:2], K + 1, return_distance=False
        )

        # connect each zdisc to K nearest neighbors
        edges = []
        for node, neighbors in enumerate(nearestNeighbors[:, 1:]):
            for neighbor in neighbors:
                edges.append((node, neighbor))
        G.add_edges_from(edges)

        return G

    def _score_graph(
        self,
        G: nx.Graph,
        c_avg_length: float = 1.0,
        c_angle: float = 1.0,
        c_length_diff: float = 1.0,
        l_avg: float = 15.0,
        l_max: float = 30.0,
    ) -> nx.Graph:
        """Assigns a score to each connection of the input graph. Higher score
        indicates the two corresponding zdiscs are likely to be two ends of a
        sarcomere.

        Parameters
        ----------
        G : nx.Graph
        c_avg_length : float, optional
            comparison of the length of a connection with ``l_avg`` affects the
            connection score, ``c_avg_length`` sets the relative effect of this
            score metric compared to the other two, by default 1.0
        c_angle : float, optional
            the angle between a connection and its neighboring connections
            affects the connection score, ``c_angle`` sets the relative effect
            of this score metric compared to the other two, by default 1.0
        c_length_diff : float, optional
            comparison of the length of a connection with the length of all
            neighboring connections affects the score, ``c_length_diff`` sets
            the relative effect of this score metric compared to the other two,
            by default 1.0
        l_avg : float, optional
            an initial guess for the average length of sarcomeres in pixels, by
            default 15.0
        l_max : float, optional
            Max allowable length for the length of sarcomeres in pixels, by
            default 30.0

        Returns
        -------
        nx.Graph
            a graph of zdiscs with all connections scored
        """
        edges_attr_dict = {}
        for node in range(G.number_of_nodes()):
            for neighbor in G.neighbors(node):
                score = 0
                v1 = G.nodes[neighbor]["pos"] - G.nodes[node]["pos"]
                l1 = np.linalg.norm(v1)
                if l1 <= l_max:
                    avg_length_score = np.exp(-np.pi * (1 - l1 / l_avg) ** 2)
                    for far_neighbor in G.neighbors(neighbor):
                        if far_neighbor in [node, neighbor]:
                            pass
                        else:
                            v2 = (
                                G.nodes[neighbor]["pos"]
                                - G.nodes[far_neighbor]["pos"]
                            )  # vector connecting neighbor to far_neighbor
                            l2 = np.linalg.norm(v2)

                            d_theta = np.arccos(np.dot(v1, v2) / (l1 * l2)) / (
                                np.pi / 2
                            )
                            d_l = np.abs(l2 - l1) / l1

                            angle_score = (
                                np.power(1 - d_theta, 2) if d_theta >= 1 else 0
                            )
                            diff_length_score = 1 / (1 + d_l)

                            score = np.max(
                                (
                                    score,
                                    c_length_diff * diff_length_score
                                    + c_angle * angle_score,
                                )
                            )
                    score += c_avg_length * avg_length_score
                edges_attr_dict[(node, neighbor)] = score

        edges_attr_dict_keep_max = {}
        for key in edges_attr_dict.keys():
            node_1 = key[0]
            node_2 = key[1]
            max_score = max(
                edges_attr_dict[(node_1, node_2)],
                edges_attr_dict[(node_2, node_1)],
            )
            edges_attr_dict_keep_max[(min(key), max(key))] = max_score
        nx.set_edge_attributes(
            G, values=edges_attr_dict_keep_max, name="score"
        )
        return G

    def _prune_graph(
        self,
        G: nx.Graph,
        score_threshold: float = 0.1,
        angle_threshold: float = 1.2,
    ) -> nx.Graph:
        """Prunes the input graph to get rid of invalid or less probable
        connections.

        Parameters
        ----------
        G : nx.Graph
            A scored graph of zdiscs clusters
        score_threshold : float
            any connection with a score less than the threshold will be
            removed, by default 0.1
        angle_threshold : float
            if a zdisc has two valid connection the angle between must be
            higher than the theshold, otherwise the connection with a lower
            score will be removed, by default 1.2 (in radians)

        Returns
        -------
        nx.Graph
        """
        nx.set_edge_attributes(G, values=0, name="validity")
        for node in range(G.number_of_nodes()):
            vectors = []
            scores = []
            neighbors = list(G.neighbors(node))
            for neighbor in neighbors:
                vectors.append(G.nodes[neighbor]["pos"] - G.nodes[node]["pos"])
                scores.append(G[node][neighbor]["score"])

            # sort by scores
            if np.max(scores) > score_threshold:
                sort_indices = np.argsort(scores)[::-1]
                best_vector = vectors[sort_indices[0]]
                G[node][neighbors[sort_indices[0]]]["validity"] += 1
                for idx in sort_indices[1:]:
                    s = scores[idx]
                    n = neighbors[idx]
                    v = vectors[idx]

                    l1 = np.linalg.norm(best_vector)
                    l2 = np.linalg.norm(v)
                    theta = np.arccos(np.dot(v, best_vector) / (l1 * l2)) / (
                        np.pi / 2
                    )

                    if theta > angle_threshold and s > score_threshold:
                        G[node][n]["validity"] += 1
                        break

        edges2remove = []
        for edge in G.edges():
            if G.edges[edge]["validity"] < 2:
                edges2remove.append(edge)
        G.remove_edges_from(edges2remove)

        return G

    def sarcomere_detection(
        self,
        file_path: str = None,
        raw_frames: np.array = None,
        segmented_zdiscs: pd.DataFrame = None,
        tracked_zdiscs: pd.DataFrame = None,
        sigma: float = 1.0,
        min_length: int = 8,
        tp_depth: float = 4.0,
        full_track_ratio: float = 0.75,
        skip_merging: bool = False,
        c_avg_length: float = 1,
        c_angle: float = 1,
        c_diff_length: float = 1,
        l_avg: float = 12.0,
        score_threshold: float = 0.01,
        angle_threshold: float = 1.2,
        save_output: bool = True,
    ) -> Tuple[pd.DataFrame, List[nx.Graph]]:
        """Detect sarcomeres in a video/image. The input could be the address
        to the video/image sample (``file_path``), raw frames as a numpy array
        (``raw_frames``), segmented zdiscs information in a pandas dataframe
        (``segmented_zdiscs``), or a dataframe of tracked zdiscs
        (``tracked_zdiscs``).

        Parameters
        ----------
        file_path : str
            The address of an image or a video file to be loaded
        raw_frames  : np.ndarray, shape=(frames, dim_1, dim_2, channels)
            Raw input image or video given as a 4 dimensional array
        segmented_zdiscs : pd.DataFrame
            Information of all detected zdiscs in every frame.
        tracked_zdiscs : pd.DataFrame
            Information of tracked zdiscs
        sigma : float
            Standard deviation for Gaussian kernel, by default ``1.0``
        min_length : int
            Minimum length for zdisc contours measured in pixels, by default
            ``8``
        tp_depth : float, optional
            the maximum distance features can move between frames, by default
            ``4.0``
        full_track_ratio : float, optional
            Frame ratio to seperate partially tracked vs fully tracked
            sarcomeres, by default ``0.75``
        skip_merging : bool, optional
            skipping the merging step will result in fewer fully tracked
            zdiscs, by default ``False``
        c_avg_length : float, optional
            comparison of the length of a connection with ``l_avg`` affects the
            connection score, ``c_avg_length`` sets the relative effect of this
            score metric compared to the other two, by default ``1.0``
        c_angle : float, optional
            the angle between a connection and its neighboring connections
            affects the connection score, ``c_angle`` sets the relative effect
            of this score metric compared to the other two, by default ``1.0``
        c_length_diff : float, optional
            comparison of the length of a connection with the length of all
            neighboring connections affects the score, ``c_length_diff`` sets
            the relative effect of this score metric compared to the other two,
            by default ``1.0``
        l_avg : float, optional
            an initial guess for the average length of sarcomeres in pixels, by
            default ``15.0``
        score_threshold : float
            any connection with a score less than the threshold will be
            removed, by default ``0.1``
        angle_threshold : float
            if a zdisc has two valid connection the angle between must be
            higher than the theshold, otherwise the connection with a lower
            score will be removed, by default ``1.2`` (in radians)
        save_output : bool
            by default ``True``

        Returns
        -------
        pd.DataFrame
            Detected sarcomeres information. Columns are ``'frame'`` (frame
            number), ``'sarc_id'`` (sarcomere id), ``'zdiscs'`` (particle id of
            the two zdiscs forming a sarcomere), ``'x'`` and ``'y'`` (sarcomere
            center position), ``'length'``, ``'width'``, and ``'angle'``
            (sarcomere length, width, and angle).
        List[nx.Graph]:
            A list of graphs each indicating connected sarcomeres (myofibrils)

        Notes
        -----
        - For a detailed description of the Trackpy package check:
          http://soft-matter.github.io/trackpy/v0.5.0/tutorial.html

        - For a detailed description of the OPTICS algorithm check:
          https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html

        See Also
        --------
        :func:`sarcgraph.sg.SarcGraph.zdisc_segmentation`

        :func:`sarcgraph.sg.SarcGraph.zdisc_tracking`
        """
        if tracked_zdiscs is None:
            tracked_zdiscs = self.zdisc_tracking(
                file_path,
                raw_frames,
                segmented_zdiscs,
                sigma,
                min_length,
                tp_depth,
                full_track_ratio,
                skip_merging,
                save_output,
            )

        zdiscs_clusters = (
            tracked_zdiscs.groupby("particle")
            .mean()
            .reset_index()[["x", "y", "particle"]]
            .to_numpy()
        )
        G = self._zdisc_to_graph(zdiscs_clusters)
        G = self._score_graph(G, c_avg_length, c_angle, c_diff_length, l_avg)
        G = self._prune_graph(G, score_threshold, angle_threshold)

        myofibrils = [G.subgraph(c).copy() for c in nx.connected_components(G)]

        sarcs = []
        num_frames = tracked_zdiscs.frame.max() + 1
        z0 = pd.DataFrame(np.arange(0, num_frames, 1), columns=["frame"])
        for i, edge in enumerate(G.edges):
            p1 = int(G.nodes[edge[0]]["particle_id"])
            p2 = int(G.nodes[edge[1]]["particle_id"])
            z1 = tracked_zdiscs[tracked_zdiscs.particle == p1]
            z2 = tracked_zdiscs[tracked_zdiscs.particle == p2]
            z1.columns = np.insert(z2.columns.values[1:] + "_p1", 0, "frame")
            z2.columns = np.insert(z2.columns.values[1:] + "_p2", 0, "frame")

            sarc = pd.merge(z0, z1, how="outer", on="frame")
            sarc = pd.merge(sarc, z2, how="outer", on="frame")

            sarc["sarc_id"] = i
            sarc["zdiscs"] = ",".join(
                map(str, sorted((p1, p2)))
            )  # list(map(float, sarc.zdiscs[0].split(',')))
            sarc["x"] = (sarc.x_p1 + sarc.x_p2) / 2
            sarc["y"] = (sarc.y_p1 + sarc.y_p2) / 2
            length = np.sqrt(
                (sarc.x_p1 - sarc.x_p2) ** 2 + (sarc.y_p1 - sarc.y_p2) ** 2
            )
            sarc["length"] = length
            width1 = np.sqrt(
                (sarc.p1_x_p1 - sarc.p2_x_p1) ** 2
                + (sarc.p1_y_p1 - sarc.p2_y_p1) ** 2
            )
            width2 = np.sqrt(
                (sarc.p1_x_p2 - sarc.p2_x_p2) ** 2
                + (sarc.p1_y_p2 - sarc.p2_y_p2) ** 2
            )
            sarc["width"] = (width1 + width2) / 2
            angle = np.arctan2(sarc.x_p2 - sarc.x_p1, sarc.y_p2 - sarc.y_p1)
            angle[angle < 0] += np.pi
            sarc["angle"] = angle
            sarcs.append(
                sarc[
                    [
                        "frame",
                        "sarc_id",
                        "x",
                        "y",
                        "length",
                        "width",
                        "angle",
                        "zdiscs",
                    ]
                ]
            )
        sarcs = pd.concat(sarcs).reset_index().drop("index", axis=1)

        if save_output:
            self._save_dataframe(sarcs, "sarcomeres")

        return sarcs, myofibrils
