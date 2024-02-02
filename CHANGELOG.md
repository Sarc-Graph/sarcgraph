# 0.3.0 - YYYY-MM-DD

## Added
- **config.py**: Introduced a new configuration file to centralize all input options and settings for the application. This file contains the `Config` class, which utilizes data classes for better structure and validation of configuration parameters. The class includes various attributes for configuration settings, along with comprehensive type annotations and validations within the `__post_init__` method.
    - Added attributes for `min_sarc_length` to allow users to set a minimum threshold for detected sarcomere length, enhancing the flexibility of sarcomere analysis.
- **sg.py**:
    - `print_config`: To display current configuration settings.
    - `_update_config`: For updating configuration with input keyword arguments.
    - `load_data`: Combines _data_loader and _to_gray, with added validity checks For loaded data dimensions.
    - `save_data`: For saving numpy arrays, lists of numpy arrays, and pandas dataframes.
    - `filter_frames` (previously `_filter_frames`): For customizable frame filtering.
    - `_validate_contours` and `_find_frame_contours`: For customizable contour detection and validation.
    - `_process_contours`: For processing all z-disc contours across frames with optional processing functions.
    - `_zdisc_center` and `_zdisc_endpoints`: Default processing functions for calculating z-disc center and endpoints.
    - `_graph_initialization`: Initializes a graph with z-discs as nodes.
    - `_find_nearest_neighbors`: Identifies the K nearest neighbors for each z-disc, with K configurable via num_neighbors.
    - `_add_edges`: Connects z-discs in the graph based on nearest neighbor relationships.
    - `_zdisc_to_graph`: Constructs a graph representing z-disc connections, linking each z-disc to its K nearest neighbors.
    - `_sarc_vector`, `_length_score`, `_sarcs_angle`, `_angle_score`, `_sarc_score`: Functions for calculating various metrics between potential sarcomeres, including length, angle, and different scorea.
    - `_score_graph`: Assigns scores to graph connections, aiding in sarcomere identification. Now allows setting minimum threshold for sarcomere length via `min_sarc_length`.
    - `_prune_graph`: Prunes connections in the graph based on scores to refine sarcomere detection.
    - `_get_connected_zdiscs`: Retrieves tracking data for z-discs connected by an edge in the graph.
    - `_initialize_sarc`: Prepares a DataFrame for sarcomere data.
    - `_process_sarc`: Processes and calculates properties of a single sarcomere.
    - `_process_sarcomeres`: Processes a graph of connected z-discs to compile comprehensive sarcomere data.

## Changed
- **sg.py**:
    - `__init__`: now optionally accepts a Config object and keyword arguments for configuration updates.
    - `_detect_contours`: to enhance customizability in contour detection. The function now exclusively takes filtered_frames as input and leverages `_find_frame_contours` and `_validate_contours` for contour detection and validation. This change allows users to define their own versions of these functions for tailored contour analysis.
    - `_validate_contours`: Modified to ignore contours detected at the edge of an image, addressing issues with improperly closed contours during analysis.
    - `zdisc_segmentation`: updated for more flexible input handling. It now accepts configuration updates via keyword arguments and supports three types of inputs: `input_file` (file address), `raw_frames` (numpy array), and `filtered_frames` (numpy array). Additionally, users can pass a list of custom processing functions to extract additional attributes from z-disc contours.
    - `_merge_tracked_zdiscs`: Adapted to accept only tracked_zdiscs as input, with adjustments for compatibility with the new Config class.
    - `zdisc_tracking`: Minor modifications implemented for compatibility with the new input option handling mechanism. Maintains core functionality while aligning with the updated approach used in `zdisc_segmentation`.
    - `sarcomere_detection`: Enhanced to align with the dynamic keyword argument handling used in `zdisc_tracking` and `zdisc_segmentation`. This function now also accepts `tracked_zdiscs` as an input, allowing for direct sarcomere detection from pre-tracked z-disc data or following z-disc tracking as needed.
- The `SarcGraph` and `SarcGraphTools` classes can now be directly imported with a simplified import statement, enhancing the usability of the package. For example, users can now use `from sarcgraph import SarcGraph` instead of the longer `from sarcgraph.sg import SarcGraph`.

## Fixed
- **sg.py**:
    - `_process_sarc`: Fixed a minor bug causing incorrect angle calculation.

### Deprecated
- **sg.py**:
    - `_data_loader`
    - `_to_gray`
    - `_save_numpy`
    - `_save_dataframe`
    - `_filter_frames`
    - `_process_input`
    - `_zdiscs_to_pandas`
