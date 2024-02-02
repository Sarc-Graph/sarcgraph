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

from sarcgraph.config import Config


class SarcGraph:
    def __init__(self, config: Config = None, **kwargs):
        """Zdiscs and sarcomeres segmentation and tracking.

        Parameters
        ----------
        config : Config, optional
            Configuration settings for the application. If not provided,
            default settings are used, and any additional settings can be
            passed via keyword arguments.
        **kwargs
            Additional settings to override the defaults or values in the
            provided Config object.
        """
        if config is None:
            config = Config()
        self.config = config

        self._update_config(**kwargs)
        self._create_output_dir()
        self.print_config()

    def print_config(self):
        self.config.print()

    def _update_config(self, **kwargs):
        """
        Update configurations with given keyword arguments.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments corresponding to configuration attributes to be
            updated.
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                if key == "output_dir":
                    self._create_output_dir()
            else:
                raise AttributeError(
                    f"{key} is not a valid configuration parameter. Use "
                    "print_config() to see all available parameters."
                )

    def _create_output_dir(self):
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def _pop_kwargs(self, *args, **kwargs):
        """Pop keys from kwargs and return the value and the updated kwargs.

        Parameters
        ----------
        *args
            Keys to pop from kwargs.
        **kwargs
            Keyword arguments to pop from.

        Returns
        -------
        tuple
            (popped_dict, updated_kwargs)
        """
        popped_dict = {key: kwargs.pop(key, None) for key in args}
        return popped_dict, kwargs

    #############################################################
    #                      Data Processing                      #
    #############################################################
    def load_data(self, file_path: str = None) -> np.ndarray:
        """Loads a video/image file.

        Parameters
        ----------
        file_path : str, optional
            A direct path to a video/image file to load.

        Returns
        -------
        numpy.ndarray
            All frames in the raw format in gray scale.
        """
        if file_path is None:
            raise ValueError("The input file_path is not specified.")

        if self.config.input_type == "video":
            try:
                data = skvideo.io.vread(file_path)
                data = np.squeeze(skvideo.utils.rgb2gray(data))
                self._check_validity(data, self.config.input_type)
            except ValueError:
                data = skimage.io.imread(file_path, plugin="tifffile")
                data = np.squeeze(skvideo.utils.rgb2gray(data))
                self._check_validity(data, self.config.input_type)
        else:
            print("Trying to load image")
            data = skimage.io.imread(file_path)
            print("got error!")
            data = np.squeeze(skvideo.utils.rgb2gray(data))
            self._check_validity(data, self.config.input_type)

        return data

    def _check_validity(self, data, input_type):
        if input_type == "image":
            if data.ndim != 2:
                raise ValueError(
                    "Loaded image data is not valid, expected a 2D image."
                )
        else:
            if not (data.ndim == 3 and data.shape[0] > 1):
                raise ValueError(
                    "Loaded video data is not valid, expected a 3D data."
                )
        return True

    def save_data(
        self, data: Union[np.ndarray, List, pd.DataFrame], file_name: str
    ) -> None:
        """Saves a numpy array or a pandas dataframe based on the type of data.

        Parameters
        ----------
        data : Union[np.ndarray, List, pd.DataFrame]
        file_name: str
        """
        if data is None:
            return

        if not isinstance(file_name, str):
            raise TypeError("file_name must be a string.")

        full_path = f"{self.config.output_dir}/{file_name}"

        if isinstance(data, (np.ndarray, list)):
            np.save(f"{full_path}.npy", data, allow_pickle=True)
        elif isinstance(data, pd.DataFrame):
            data.to_csv(f"{full_path}.csv")
        else:
            raise TypeError(
                "data must be either a numpy.ndarray, a list, or a"
                " pandas.DataFrame."
            )

    def filter_frames(self, frames: np.ndarray) -> np.ndarray:
        """Convolves each image with Laplacian and Gaussian filters.

        Parameters
        ----------
        frames : np.ndarray
            A 2D image or a stack of 2D images.

        Returns
        -------
        np.ndarray
            A stack of filtered images.
        """
        sigma = self.config.sigma

        if frames.ndim == 2:
            frames = frames[np.newaxis, ...]
        elif frames.ndim != 3:
            raise ValueError(
                "Input must be a 2D image or a stack of 2D images."
            )

        # Initialize an array to hold the filtered images
        filtered_frames = np.zeros_like(frames, dtype=np.float64)

        # Apply filters to each frame
        for i in range(frames.shape[0]):
            frame = frames[i]
            # Apply the Laplacian filter
            laplacian_filtered = laplace(frame)
            # Apply the Gaussian filter
            gaussian_filtered = gaussian(laplacian_filtered, sigma=sigma)
            filtered_frames[i] = gaussian_filtered
        return filtered_frames

    #############################################################
    #                    Z-Disc Segmentation                    #
    #############################################################
    def zdisc_segmentation(self, **kwargs):
        """
        Perform z-disc segmentation using various inputs and optional
        processing functions.
        - Pre-filtered frames can be provided by specifying 'filtered_frames'.
        - Raw frames can be provided by specifying 'raw_frames'; overrides
        'filtered_frames'.
        - An input file path can be provided by specifying 'input_file';
        overrides 'raw_frames' and 'filtered_frames'.
        - Processing functions can be provided as a list of callable functions
        by specifying 'processing_functions'.

        Any configuration parameters can also be updated to customize the
        behavior of the segmentation and processing.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments including:
            - 'input_file': Path to the input file to load raw frames.
            - 'raw_frames': Array-like structure with raw frames for
            processing.
            - 'filtered_frames': Array-like structure with pre-filtered frames
            for processing.
            - 'processing_functions': List of function references to apply
            additional processing.
            - Any configuration parameters to update.

        Returns
        -------
        pd.DataFrame
            The processed z-disc information after segmentation.
        """
        # Extract specific kwargs for input data
        inputs_dict, kwargs = self._pop_kwargs(
            "input_file",
            "raw_frames",
            "filtered_frames",
            "processing_functions",
            **kwargs,
        )
        input_file = inputs_dict["input_file"]
        raw_frames = inputs_dict["raw_frames"]
        filtered_frames = inputs_dict["filtered_frames"]
        processing_functions = []
        if inputs_dict["processing_functions"] is not None:
            processing_functions = inputs_dict["processing_functions"]

        self._update_config(**kwargs)

        # Validate provided processing functions
        if not all(callable(fn) for fn in processing_functions):
            raise ValueError(
                "All items in 'processing_functions' must be ", "callable."
            )

        mock_contour_input = np.array(
            [[0, 1], [1, 0], [2, 1], [2, 2], [1, 3], [0, 2], [0, 1]]
        )
        for fn in processing_functions:
            test_output = fn(mock_contour_input)
            if not isinstance(test_output, dict):
                raise ValueError(
                    f"The function {fn.__name__} did not return a dictionary."
                )

        # Load data if 'input_file' is provided
        if input_file is not None:
            raw_frames = self.load_data(input_file)

        # Generate filtered frames if 'raw_frames' are provided
        if raw_frames is not None:
            filtered_frames = self.filter_frames(raw_frames)

        # If no 'filtered_frames' are provided by now, raise an error
        if filtered_frames is None:
            raise ValueError(
                "No valid input data provided. Please specify "
                "'input_file', 'raw_frames', or 'filtered_frames'"
                "."
            )

        # Detect contours from filtered frames
        contours = self._detect_contours(filtered_frames)

        # Default processing functions
        default_processing_methods = [
            self._zdisc_center,
            self._zdisc_endpoints,
        ]

        # Apply additional processing functions provided by the user
        zdiscs = self._process_contours(
            contours, default_processing_methods + processing_functions
        )

        # Save output if configured to do so
        if self.config.save_output:
            self.save_data(raw_frames, "raw_frames")
            self.save_data(filtered_frames, "filtered_frames")
            self.save_data(contours, "contours")
            self.save_data(zdiscs, "segmented_zdiscs")

        return zdiscs

    def _detect_contours(
        self, filtered_frames: np.ndarray
    ) -> List[np.ndarray]:
        """Returns contours of detected zdiscs in all frames filtered by the
        length threshold specified in the configuration.

        Parameters
        ----------
        filtered_frames : np.ndarray

        Returns
        -------
        np.ndarray of shape (num_frames, num_contours, contour_length, 2)
        """
        if filtered_frames.ndim != 3:
            raise ValueError(
                "The input must be a 3D numpy array: (frames, " "dim_1, dim_2)"
            )

        valid_contours = []
        for frame in filtered_frames:
            contours = self._find_frame_contours(frame)
            valid_contours_for_frame = self._validate_contours(contours)
            valid_contours.append(valid_contours_for_frame)

        return np.array(valid_contours, dtype=object)

    def _validate_contours(
        self, contours: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Validates contours based on their length and ensure they are closed.

        Parameters
        ----------
        contours : List[np.ndarray]
            List of numpy arrays of detected contours.

        Returns
        -------
        List[np.ndarray]
            List of numpy arrays of valid contours.
        """
        valid_contours = []
        for contour in contours:
            if not np.allclose(contour[0], contour[-1]):
                continue
            if (
                self.config.zdisc_min_length <= len(contour)
                and len(contour) <= self.config.zdisc_max_length
            ):
                valid_contours.append(contour)

        return valid_contours

    def _find_frame_contours(self, frame: np.ndarray) -> List[np.ndarray]:
        """Detects contours within a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Single frame of the filtered image stack.

        Returns
        -------
        List[np.array]
            List of numpy arrays representing detected contours.
        """
        contour_thresh = threshold_otsu(frame)
        contours = measure.find_contours(frame, contour_thresh)
        return contours

    def _process_contours(
        self,
        contours_all: Union[List, np.ndarray],
        processing_functions: List[callable],
    ) -> pd.DataFrame:
        """
        Processes a list of contours across all frames using a list of default
        and user-defined functions and saves the results as a dataframe

        Parameters
        ----------
        contours_all : List or np.ndarray
            A list of z-discs contours as 2D numpy arrays.
        processing_functions : List[callable]
            A list of functions that each take a contour as input and return a
            dictionary where keys are attribute names and values are the
            corresponding data (e.g. {}'x': 125.64}).

        Returns
        -------
        pd.DataFrame
            A dataframe where each row represents a zdisc and includes data
            from all processing functions.
        """
        data_frame_list = []
        for frame_index, contours in enumerate(contours_all):
            for contour in contours:
                zdisc_info = {"frame": frame_index}
                for func in processing_functions:
                    zdisc_info.update(func(contour))
                data_frame_list.append(zdisc_info)
        return pd.DataFrame(data_frame_list)

    def _zdisc_center(self, contour: np.ndarray) -> dict:
        """
        Calculate the centroid of a zdisc given its contour.

        Parameters
        ----------
        contour : np.ndarray
            NumPy array of shape (contour_length, 2) representing z-disc

        Returns
        -------
        dict
            A dictionary with keys 'x' and 'y' representing the z-disc centroid
        """
        center_coords = np.mean(np.unique(contour, axis=0), axis=0)
        return {"x": center_coords[0], "y": center_coords[1]}

    def _zdisc_endpoints(self, contour: Union[List, np.ndarray]) -> dict:
        """
        Identify the main axis of a zdisc by finding the two points in the
        contour that are farthest apart from each other.

        Parameters
        ----------
        contour : np.ndarray
            NumPy array of shape (contour_length, 2) representing z-disc

        Returns
        -------
        dict
            A dictionary with keys 'p1_x', 'p1_y', 'p2_x', 'p2_y' representing
            the coordinates of the two endpoints.
        """
        dist_mat = distance_matrix(contour, contour)
        indices = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
        p1, p2 = contour[indices[0]], contour[indices[1]]
        return {"p1_x": p1[0], "p1_y": p1[1], "p2_x": p2[0], "p2_y": p2[1]}

    #########################################################
    #                    Z-disc Tracking                    #
    #########################################################
    def _merge_tracked_zdiscs(
        self,
        tracked_zdiscs: pd.DataFrame,
    ) -> pd.DataFrame:
        """A post processing step to group related partially tracked zdiscs
        using the OPTICS algorithm. Increases the robustness of zdisc tracking
        as well as the number of fully tracked zdiscs.

        Parameters
        ----------
        tracked_zdiscs : pd.DataFrame
            tracked zdiscs information for all frames

        Returns
        -------
        pd.DataFrame

        Notes
        -----
        For a detailed description of the OPTICS algorithm check:
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html
        """
        full_track_ratio = self.config.full_track_ratio
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

    def zdisc_tracking(self, **kwargs) -> pd.DataFrame:
        """Track detected Z-Discs in video data, similar to zdisc_segmentation,
        but with an additional tracking step. If 'segmented_zdiscs' is
        provided, it directly proceeds to tracking; otherwise, it first runs
        the z-discs segmentation process.

        This function shares similar inputs to `zdisc_segmentation` with the
        addition of an optional 'segmented_zdiscs' DataFrame parameter.

        Parameters
        ----------
        **kwargs : dict, optional
            Shared keyword arguments with `zdisc_segmentation`, plus:
            - segmented_zdiscs: pd.DataFrame, optional
            Pre-segmented zdiscs information. If provided, it should contain
            at least the following columns: 'frame', 'x', 'y', 'p1_x', 'p1_y',
            'p2_x', 'p2_y'.

        Returns
        -------
        pd.DataFrame
            Tracked zdiscs information; adds 'particle' column for each tracked
            zdisc.

        Notes
        -----
        - For details on shared parameters and segmentation process, refer to
        :func:`sarcgraph.sg.SarcGraph.zdisc_segmentation`.

        See Also
        --------
        :func:`sarcgraph.sg.SarcGraph.zdisc_segmentation`
        """
        # Extract specific kwargs for input data
        inputs_dict, kwargs = self._pop_kwargs("segmented_zdiscs", **kwargs)
        segmented_zdiscs = inputs_dict["segmented_zdiscs"]

        if segmented_zdiscs is None:
            detected_zdiscs = self.zdisc_segmentation(**kwargs)
            return self.zdisc_tracking(
                segmented_zdiscs=detected_zdiscs, **kwargs
            )

        # Validate segmented_zdiscs if provided
        required_columns = {
            "frame",
            "x",
            "y",
            "p1_x",
            "p1_y",
            "p2_x",
            "p2_y",
        }
        if (
            not isinstance(segmented_zdiscs, pd.DataFrame)
            or segmented_zdiscs.empty
            or not required_columns.issubset(segmented_zdiscs.columns)
        ):
            raise ValueError(
                "Provided 'segmented_zdiscs' DataFrame is not in the "
                "correct format. The DataFrame must be non-empty and "
                "include the following columns: 'frame', 'x', 'y', 'p1_x',"
                " 'p1_y', 'p2_x', 'p2_y'."
            )

        _, kwargs = self._pop_kwargs(
            "input_file", "raw_frames", "filtered_frames", **kwargs
        )
        self._update_config(**kwargs)

        if self.config.input_type == "image":
            segmented_zdiscs["particle"] = np.arange(len(segmented_zdiscs))
            tracked_zdiscs = segmented_zdiscs
        else:
            num_frames = len(segmented_zdiscs["frame"].unique())
            t = tp.link_df(
                segmented_zdiscs,
                search_range=self.config.tp_depth,
                memory=num_frames,
            )
            tracked_zdiscs = tp.filter_stubs(t, 2).reset_index(drop=True)
            if not self.config.skip_merge:
                tracked_zdiscs = self._merge_tracked_zdiscs(tracked_zdiscs)
        if self.config.save_output:
            self.save_data(tracked_zdiscs, "tracked_zdiscs")

        return tracked_zdiscs

    #############################################################
    #                    Sarcomere Detection                    #
    #############################################################
    def sarcomere_detection(
        self, **kwargs
    ) -> Tuple[pd.DataFrame, List[nx.Graph]]:
        """Detect sarcomeres in a video/image using dynamic keyword arguments.
        Sarcomere detection can be initiated with tracked zdisc information,
        or by first running zdisc tracking.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments dynamically passed. Relevant keys are:
            'input_file' : str
                The address of an image or a video file to be loaded.
            'raw_frames' : np.ndarray
                Raw input frames as a numpy array.
            'filtered_frames' : np.ndarray
                Pre-filtered frames for processing.
            'segmented_zdiscs' : pd.DataFrame
                Information of all detected zdiscs in every frame.
            'tracked_zdiscs' : pd.DataFrame
                Information of tracked zdiscs. Must be non-empty and include
                the following columns: 'frame', 'x', 'y', 'p1_x', 'p1_y',
                'p2_x', 'p2_y', 'particle'. If provided and valid, sarcomere
                detection is initiated immediately.

            Additional configuration parameters relevant for sarcomere
            detection and tracking can also be provided via kwargs.

        Returns
        -------
        Tuple[pd.DataFrame, List[nx.Graph]]
            A tuple containing:
            - A DataFrame with detected sarcomeres information including
              'frame', 'sarc_id', 'zdiscs', 'x', 'y', 'length', 'width', and
              'angle'.
            - A list of Graph objects representing connected sarcomeres
              (myofibrils).

        Notes
        -----
        - If 'tracked_zdiscs' is not provided, this function will internally
          call `zdisc_tracking` to generate tracked zdiscs from the provided
          keyword arguments. It is assumed that `zdisc_tracking` can handle
          error checking and appropriate defaults for its parameters.

        - For a detailed description of the Trackpy package check:
          http://soft-matter.github.io/trackpy/v0.5.0/tutorial.html

        - For a detailed description of the OPTICS algorithm check:
          https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html

        See Also
        --------
        :func:`sarcgraph.sg.SarcGraph.zdisc_segmentation`
        :func:`sarcgraph.sg.SarcGraph.zdisc_tracking`
        """
        # Extract specific kwargs for input data
        inputs_dict, kwargs = self._pop_kwargs("tracked_zdiscs", **kwargs)
        tracked_zdiscs = inputs_dict["tracked_zdiscs"]

        if tracked_zdiscs is None:
            tracked_zdiscs = self.zdisc_tracking(**kwargs)
            return self.sarcomere_detection(
                tracked_zdiscs=tracked_zdiscs, **kwargs
            )

        # Validate segmented_zdiscs if provided
        required_columns = {
            "frame",
            "x",
            "y",
            "p1_x",
            "p1_y",
            "p2_x",
            "p2_y",
            "particle",
        }
        if (
            not isinstance(tracked_zdiscs, pd.DataFrame)
            or tracked_zdiscs.empty
            or not required_columns.issubset(tracked_zdiscs.columns)
        ):
            raise ValueError(
                "Provided 'tracked_zdiscs' DataFrame is not in the correct "
                "format. The DataFrame must be non-empty and include at least "
                "the following columns: 'frame', 'x', 'y', 'p1_x', 'p1_y', "
                "'p2_x', 'p2_y', 'particle'."
            )

        _, kwargs = self._pop_kwargs(
            "input_file",
            "raw_frames",
            "filtered_frames",
            "segmented_zdiscs",
            **kwargs,
        )

        self._update_config(**kwargs)

        zdiscs_clusters = (
            tracked_zdiscs.groupby("particle")
            .mean()
            .reset_index()[["x", "y", "particle"]]
            .to_numpy()
        )
        G = self._zdisc_to_graph(zdiscs_clusters)
        G = self._score_graph(G)
        G = self._prune_graph(G)

        myofibrils = [G.subgraph(c).copy() for c in nx.connected_components(G)]

        sarcs = self._process_sarcomeres(G, tracked_zdiscs)

        if self.config.save_output:
            self.save_data(sarcs, "sarcomeres")

        return sarcs, myofibrils

    def _zdisc_to_graph(self, zdiscs: np.array) -> nx.Graph:
        """Creates a graph with zdiscs as nodes. Each zdisc is connected to its
        ``K`` nearest neighbors.

        Parameters
        ----------
        zdiscs : np.array, shape=(N, 3)
            zdiscs information as an array. The first two columns are the x and
            y location of zdisc centers and the last is the particle id.

        Returns
        -------
        nx.Graph
        """
        G = self._graph_initialization(zdiscs)
        return self._add_edges(G, self._find_nearest_neighbors(zdiscs[:, 0:2]))

    def _graph_initialization(self, zdiscs: np.array) -> nx.Graph:
        """Initializes a graph of z-discs.

        Parameters
        ----------
        zdiscs : np.array
            An array of z-disc data. The first two columns are expected to be x
            and y coordinates, and the last column is expected to be the
            particle ID.

        Returns
        -------
        nx.Graph
            A networkx Graph object where each node represents a z-disc and has
            'pos' (position) and 'particle_id' attributes.
        """
        num_nodes = len(zdiscs)
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        nodes_pos_dict = {i: pos for i, pos in enumerate(zdiscs[:, 0:2])}
        nodes_particle_dict = {j: id for j, id in enumerate(zdiscs[:, -1])}

        nx.set_node_attributes(G, values=nodes_pos_dict, name="pos")
        nx.set_node_attributes(
            G, values=nodes_particle_dict, name="particle_id"
        )

        return G

    def _find_nearest_neighbors(self, zdiscs: np.array) -> np.array:
        """Finds the K nearest neighbors of each z-disc. K can be specified as
        a config parameter num_neighbors.

        Parameters
        ----------
        zdiscs : np.array
            Array of z-disc positions (x and y).

        Returns
        -------
        np.array
            Array of indices of nearest neighbors for each z-disc.
        """
        K = self.config.num_neighbors
        neigh = NearestNeighbors(n_neighbors=2)
        neigh.fit(zdiscs)
        nearestNeighbors = neigh.kneighbors(
            zdiscs, K + 1, return_distance=False
        )
        return nearestNeighbors

    def _add_edges(self, G: nx.Graph, nearestNeighbors: np.array) -> nx.Graph:
        """Adds edges to the graph by connecting on K nearest neighbors.

        Parameters
        ----------
        G : nx.Graph
            The graph to which edges will be added.
        nearestNeighbors : np.array
            Array of indices of nearest neighbors.

        Returns
        -------
        nx.Graph
            The graph with added edges.
        """
        edges = []
        for node, neighbors in enumerate(nearestNeighbors[:, 1:]):
            for neighbor in neighbors:
                edges.append((node, neighbor))
        G.add_edges_from(edges)

        return G

    def _score_graph(self, G: nx.Graph) -> nx.Graph:
        """Assigns a score to each connection of the input graph. Higher score
        indicates the two corresponding zdiscs are likely to be two ends of a
        sarcomere.

        Parameters
        ----------
        G : nx.Graph

        Returns
        -------
        nx.Graph
            a graph of zdiscs with all connections scored
        """
        c_avg_length = self.config.coeff_avg_length
        l_avg = self.config.avg_sarc_length
        l_max = self.config.max_sarc_length
        l_min = self.config.min_sarc_length
        edges_attr_dict = {}
        for node in range(G.number_of_nodes()):
            for neighbor in G.neighbors(node):
                score = 0
                v1, l1 = self._sarc_vector(G, node, neighbor)
                if l1 <= l_max and l1 >= l_min:
                    avg_length_score = np.exp(-np.pi * (1 - l1 / l_avg) ** 2)
                    for far_neighbor in G.neighbors(neighbor):
                        if far_neighbor in [node, neighbor]:
                            pass
                        else:
                            v2, l2 = self._sarc_vector(
                                G, far_neighbor, neighbor
                            )
                            sum_scores = self._sarc_score(v1, v2, l1, l2)
                            score = np.max((score, sum_scores))
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

    def _sarc_vector(self, G, node1, node2):
        """Calculates the vector and length between two nodes in the graph.

        Parameters
        ----------
        G : nx.Graph
        node1, node2
            The node ids for which the vector is calculated.

        Returns
        -------
        tuple
            The vector connecting node1 to node2 and its length.
        """
        sarc = G.nodes[node2]["pos"] - G.nodes[node1]["pos"]
        length = np.linalg.norm(sarc)
        return sarc, length

    def _length_score(self, l1, l2):
        """Calculates the length score between two potential sarcomeres.

        Parameters
        ----------
        l1, l2
            The lengths of the two sarcomeres to compare.

        Returns
        -------
        float
            The calculated length score.
        """
        d_l = np.abs(l2 - l1) / l1
        return 1 / (1 + d_l)

    def _sarcs_angle(self, v1, v2, l1, l2):
        """Calculates the angle between two vectors.

        Parameters
        ----------
        v1, v2 : np.array
        l1, l2 : float

        Returns
        -------
        float
        The angle between the two vectors.
        """
        return np.arccos(np.dot(v1, v2) / (l1 * l2)) / (np.pi / 2)

    def _angle_score(self, v1, v2, l1, l2):
        """
        Calculates the angle score between two potential sarcomeres.

        Parameters
        ----------
        v1, v2 : np.array
        l1, l2 : float

        Returns
        -------
        float
            The calculated angle score.
        """
        theta = self._sarcs_angle(v1, v2, l1, l2)
        return np.power(theta - 1, 2) if theta >= 1 else 0

    def _sarc_score(self, v1, v2, l1, l2):
        """
        Calculates sarcomere score based on length and angle scores.

        Parameters
        ----------
        v1, v2 : np.array
            The vectors representing two potential connected sarcomeres.
        l1, l2 : float
            The lengths of the two sarcomeres.

        Returns
        -------
        float
            The sarcomere score.
        """
        c_len = self.config.coeff_neighbor_length
        c_ang = self.config.coeff_neighbor_angle
        len_score = self._length_score(l1, l2)
        ang_score = self._angle_score(v1, v2, l1, l2)
        return c_len * len_score + c_ang * ang_score

    def _prune_graph(self, G: nx.Graph) -> nx.Graph:
        """Prunes the input graph to get rid of invalid or less probable
        connections.

        Parameters
        ----------
        G : nx.Graph
            A scored graph of zdiscs clusters

        Returns
        -------
        nx.Graph
        """
        score_threshold = self.config.score_threshold
        angle_threshold = self.config.angle_threshold
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
                    theta = self._sarcs_angle(v, best_vector, l1, l2)

                    if theta > angle_threshold and s > score_threshold:
                        G[node][n]["validity"] += 1
                        break

        edges2remove = []
        for edge in G.edges():
            if G.edges[edge]["validity"] < 2:
                edges2remove.append(edge)
        G.remove_edges_from(edges2remove)

        return G

    def _get_connected_zdiscs(
        self,
        G: nx.Graph,
        tracked_zdiscs: pd.DataFrame,
        edge: Tuple[int, int],
    ) -> [pd.DataFrame, pd.DataFrame]:
        """
        Retrieves the tracking data for the z-discs connected by a given edge
        in the graph.

        Parameters
        ----------
        G : nx.Graph
            The graph representing z-disc connections.
        tracked_zdiscs : pd.DataFrame
            The dataframe with tracking data for all z-discs.
        edge : tuple
            The tuple representing the edge connecting two z-discs in the
            graph.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple of dataframes, each corresponding to the tracking data of
            one of the two z-discs connected by the edge.
        """
        p1 = int(G.nodes[edge[0]]["particle_id"])
        p2 = int(G.nodes[edge[1]]["particle_id"])
        z1 = tracked_zdiscs[tracked_zdiscs.particle == p1]
        z2 = tracked_zdiscs[tracked_zdiscs.particle == p2]
        z1.columns = np.insert(z1.columns.values[1:] + "_p1", 0, "frame")
        z2.columns = np.insert(z2.columns.values[1:] + "_p2", 0, "frame")

        return z1, z2

    def _initialize_sarc(
        self, z0: pd.DataFrame, z1: pd.DataFrame, z2: pd.DataFrame
    ) -> pd.DataFrame:
        """
        This function creates a base DataFrame for a single sarcomere by
        merging base frame information from `z0` with z-disc data from `z1` and
        `z2`.

        Parameters
        ----------
        z0 : pd.DataFrame
            A DataFrame with a single column 'frame' representing all frames in
            the dataset.
        z1 : pd.DataFrame
            A DataFrame containing the details of the first z-disc of the
            sarcomere in each frame where it is present.
        z2 : pd.DataFrame
            A DataFrame containing the details of the second z-disc of the
            sarcomere in each frame where it is present.

        Returns
        -------
        pd.DataFrame
            A merged DataFrame representing the initial state of a single
            sarcomere across all frames.
        """
        sarc = pd.merge(z0, z1, how="outer", on="frame")
        sarc = pd.merge(sarc, z2, how="outer", on="frame")

        return sarc

    def _process_sarc(self, sarc: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a single sarcomere's data to calculate its properties.

        Takes a dataframe initialized with sarcomere information and computes
        various properties such as center position, length, width, and angle
        based on the positions of the connected z-discs.

        Parameters
        ----------
        sarc : pd.DataFrame
            The initialized dataframe for a single sarcomere containing the
            positions of the connected z-discs and other relevant information.

        Returns
        -------
        pd.DataFrame
            The dataframe for a single sarcomere with additional computed
            properties.
        """
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
        angle = np.pi - angle
        sarc["angle"] = angle
        sarc = sarc[
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
        return sarc

    def _process_sarcomeres(
        self, G: nx.Graph, tracked_zdiscs: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Processes sarcomeres from a graph of connected z-discs.

        This function takes a graph where nodes represent z-discs and edges
        represent connections between them, alongside a dataframe of tracked
        z-discs across frames. It combines information from connected z-discs,
        and compiles processed sarcomere properties into a dataframe.

        Parameters
        ----------
        G : nx.Graph
            A networkx Graph object where nodes represent z-discs and edges
            represent sarcomeres connecting these z-discs.
        tracked_zdiscs : pd.DataFrame
            A pandas dataframe containing tracked positions of z-discs across
            different frames.

        Returns
        -------
        pd.DataFrame
            A dataframe containing processed sarcomere data with assigned
            identifiers and their properties such as length, width, and angle
            for each frame.
        """
        sarcs = []
        num_frames = tracked_zdiscs.frame.max() + 1
        z0 = pd.DataFrame(np.arange(0, num_frames, 1), columns=["frame"])
        for i, edge in enumerate(G.edges):
            z1, z2 = self._get_connected_zdiscs(G, tracked_zdiscs, edge)
            sarc = self._initialize_sarc(z0, z1, z2)
            sarc["sarc_id"] = i
            p1 = z1.particle_p1.values[0]
            p2 = z2.particle_p2.values[0]
            sarc["zdiscs"] = ",".join(map(str, sorted((p1, p2))))
            sarcs.append(self._process_sarc(sarc))
        sarcs = pd.concat(sarcs).reset_index().drop("index", axis=1)

        return sarcs
