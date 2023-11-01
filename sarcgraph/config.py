from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """
    Configuration settings for the application.

    :ivar output_dir: Directory to save output files. Defaults to "output".
    :type output_dir: str
    :ivar input_type: Type of the input data, either "image" or "video".
    Defaults to "video".
    :type input_type: str
    :ivar input_file: Path to the input file. Defaults to None.
    :type input_file: Optional[str]
    :ivar save_output: Flag to determine whether to save the output file.
    Defaults to True.
    :type save_output: bool
    :ivar gaussian_sigma: Standard deviation for Gaussian filter. Defaults to
    1.0.
    :type gaussian_sigma: float
    :ivar zdisc_min_length: Minimum length for Z-disc contours. Defaults to 15.
    :type zdisc_min_length: int
    :ivar zdisc_max_length: Maximum length for Z-disc contours. Defaults to 50.
    :type zdisc_max_length: int
    :ivar full_track_ratio: Ratio of frames required to consider a Z-disc as
    fully tracked. Defaults to 0.75.
    :type full_track_ratio: float
    :ivar tp_depth: TrackPy's depth parameter. Defaults to 4.
    :type tp_depth: int
    :ivar skip_merge: Flag to skip post-processing merging step of partially
    tracked Z-discs. Defaults to False.
    :type skip_merge: bool
    :ivar num_neighbors: Number of nearest neighbors for each Z-disc during
    graph construction. Defaults to 3.
    :type num_neighbors: int
    :ivar avg_sarc_length: Estimated average sarcomere length. Defaults to
    15.0.
    :type avg_sarc_length: float
    :ivar max_sarc_length: Maximum allowed sarcomere length. Defaults to 30.0.
    :type max_sarc_length: float
    :ivar coeff_avg_length: Sarcomere scoring coefficient for average length
    difference. Defaults to 1.0.
    :type coeff_avg_length: float
    :ivar coeff_neighbor_length: Sarcomere scoring coefficient for neighbor
    length difference. Defaults to 1.0.
    :type coeff_neighbor_length: float
    :ivar coeff_neighbor_angle: Sarcomere scoring coefficient for neighbor
    angle difference. Defaults to 1.0.
    :type coeff_neighbor_angle: float
    :ivar score_threshold: Threshold for valid sarcomere score. Defaults to
    0.1.
    :type score_threshold: float
    :ivar angle_threshold: Minimum allowed angle between connected sarcomeres.
    Defaults to 1.2.
    :type angle_threshold: float
    """

    output_dir: str = "output"
    input_type: str = "video"
    input_file: Optional[str] = None
    save_output: bool = True
    gaussian_sigma: float = 1.0
    zdisc_min_length: int = 15
    zdisc_max_length: int = 50
    full_track_ratio: float = 0.75
    tp_depth: int = 4
    skip_merge: bool = False
    num_neighbors: int = 3
    avg_sarc_length: float = 15.0
    max_sarc_length: float = 30.0
    coeff_avg_length: float = 1.0
    coeff_neighbor_length: float = 1.0
    coeff_neighbor_angle: float = 1.0
    score_threshold: float = 0.1
    angle_threshold: float = 1.2

    VALID_INPUT_TYPES = ["image", "video"]

    def __post_init__(self):
        # Type checks
        if not isinstance(self.output_dir, str):
            raise TypeError("output_dir must be a string")
        if not isinstance(self.input_type, str):
            raise TypeError("input_type must be a string")
        if not (self.input_file is None or isinstance(self.input_file, str)):
            raise TypeError("input_file must be a string or None")
        if not isinstance(self.save_output, bool):
            raise TypeError("save_output must be a boolean")
        if not isinstance(self.gaussian_sigma, float):
            raise TypeError("gaussian_sigma must be a float")
        if not isinstance(self.zdisc_min_length, int):
            raise TypeError("zdisc_min_length must be an integer")
        if not isinstance(self.zdisc_max_length, int):
            raise TypeError("zdisc_max_length must be an integer")
        if not isinstance(self.full_track_ratio, float):
            raise TypeError("full_track_ratio must be a float")
        if not isinstance(self.tp_depth, int):
            raise TypeError("tp_depth must be an integer")
        if not isinstance(self.skip_merge, bool):
            raise TypeError("skip_merge must be a boolean")
        if not isinstance(self.num_neighbors, int):
            raise TypeError("num_neighbors must be an integer")
        if not isinstance(self.avg_sarc_length, float):
            raise TypeError("avg_sarc_length must be a float")
        if not isinstance(self.max_sarc_length, float):
            raise TypeError("max_sarc_length must be a float")
        if not isinstance(self.coeff_avg_length, float):
            raise TypeError("coeff_avg_length must be a float")
        if not isinstance(self.coeff_neighbor_length, float):
            raise TypeError("coeff_neighbor_length must be a float")
        if not isinstance(self.coeff_neighbor_angle, float):
            raise TypeError("coeff_neighbor_angle must be a float")
        if not isinstance(self.score_threshold, float):
            raise TypeError("score_threshold must be a float")
        if not isinstance(self.angle_threshold, float):
            raise TypeError("angle_threshold must be a float")

        # Value checks
        if self.input_type not in self.VALID_INPUT_TYPES:
            raise ValueError("input_type must be one of "
                             f"{self.VALID_INPUT_TYPES}")
        if self.zdisc_max_length <= self.zdisc_min_length:
            raise ValueError("zdisc_max_length must be greater than "
                             "zdisc_min_length")
        if not 0.5 <= self.full_track_ratio <= 1.0:
            raise ValueError("full_track_ratio must be between 0.5 and 1.0")
        if self.max_sarc_length <= self.avg_sarc_length:
            raise ValueError("max_sarc_length must be greater than "
                             "avg_sarc_length")
        if self.score_threshold <= 0.0:
            raise ValueError("score_threshold must be greater than 0.0")
        if self.angle_threshold <= 1.0:
            raise ValueError("angle_threshold must be greater than 1.0")
