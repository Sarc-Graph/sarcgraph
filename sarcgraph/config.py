from dataclasses import dataclass, field
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
    _output_dir: str = field(default="output", init=False)
    _input_type: str = field(default="video", init=False)
    _input_file: Optional[str] = field(default=None, init=False)
    _save_output: bool = field(default=True, init=False)
    _gaussian_sigma: float = field(default=1.0, init=False)
    _zdisc_min_length: int = field(default=15, init=False)
    _zdisc_max_length: int = field(default=50, init=False)
    _full_track_ratio: float = field(default=0.75, init=False)
    _tp_depth: int = field(default=4, init=False)
    _skip_merge: bool = field(default=False, init=False)
    _num_neighbors: int = field(default=3, init=False)
    _avg_sarc_length: float = field(default=15.0, init=False)
    _max_sarc_length: float = field(default=30.0, init=False)
    _coeff_avg_length: float = field(default=1.0, init=False)
    _coeff_neighbor_length: float = field(default=1.0, init=False)
    _coeff_neighbor_angle: float = field(default=1.0, init=False)
    _score_threshold: float = field(default=0.1, init=False)
    _angle_threshold: float = field(default=1.2, init=False)

    VALID_INPUT_TYPES = ["image", "video"]

    def __post_init__(self):
        self.output_dir = self._output_dir
        self.input_type = self._input_type
        self.input_file = self._input_file
        self.save_output = self._save_output
        self.gaussian_sigma = self._gaussian_sigma
        self.zdisc_min_length = self._zdisc_min_length
        self.zdisc_max_length = self._zdisc_max_length
        self.full_track_ratio = self._full_track_ratio
        self.tp_depth = self._tp_depth
        self.skip_merge = self._skip_merge
        self.num_neighbors = self._num_neighbors
        self.avg_sarc_length = self._avg_sarc_length
        self.max_sarc_length = self._max_sarc_length
        self.coeff_avg_length = self._coeff_avg_length
        self.coeff_neighbor_length = self._coeff_neighbor_length
        self.coeff_neighbor_angle = self._coeff_neighbor_angle
        self.score_threshold = self._score_threshold
        self.angle_threshold = self._angle_threshold

    @property
    def output_dir(self) -> str:
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value: str):
        if not isinstance(value, str):
            raise TypeError("output_dir must be a string")
        self._output_dir = value

    @property
    def input_type(self) -> str:
        return self._input_type

    @input_type.setter
    def input_type(self, value: str):
        if not isinstance(value, str):
            raise TypeError("input_type must be a string")
        self._input_type = value

    @property
    def input_file(self) -> Optional[str]:
        return self._input_file

    @input_file.setter
    def input_file(self, value: Optional[str]):
        if value is not None and not isinstance(value, str):
            raise TypeError("input_file must be a string")
        self._input_file = value

    @property
    def save_output(self) -> bool:
        return self._save_output

    @save_output.setter
    def save_output(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("save_output must be a boolean")
        self._save_output = value

    @property
    def gaussian_sigma(self) -> float:
        return self._gaussian_sigma

    @gaussian_sigma.setter
    def gaussian_sigma(self, value: float):
        if not isinstance(value, float):
            raise TypeError("gaussian_sigma must be a float")
        if value <= 0.0:
            raise ValueError("gaussian_sigma must be greater than 0.0")
        self._gaussian_sigma = value

    @property
    def zdisc_min_length(self) -> int:
        return self._zdisc_min_length

    @zdisc_min_length.setter
    def zdisc_min_length(self, value: int):
        if not isinstance(value, int):
            raise TypeError("zdisc_min_length must be an integer")
        if value <= 2:
            raise ValueError("zdisc_min_length must be greater than 2")
        self._zdisc_min_length = value

    @property
    def zdisc_max_length(self) -> int:
        return self._zdisc_max_length

    @zdisc_max_length.setter
    def zdisc_max_length(self, value: int):
        if not isinstance(value, int):
            raise TypeError("zdisc_max_length must be an integer")
        if value <= self.zdisc_min_length:
            raise ValueError("zdisc_max_length must be greater than "
                             "zdisc_min_length")
        self._zdisc_max_length = value

    @property
    def full_track_ratio(self) -> float:
        return self._full_track_ratio

    @full_track_ratio.setter
    def full_track_ratio(self, value: float):
        if not isinstance(value, float):
            raise TypeError("full_track_ratio must be a float")
        if not 0.5 <= value <= 1.0:
            raise ValueError("full_track_ratio must be between 0.5 and 1.0")
        self._full_track_ratio = value

    @property
    def tp_depth(self) -> int:
        return self._tp_depth

    @tp_depth.setter
    def tp_depth(self, value: int):
        if not isinstance(value, int):
            raise TypeError("tp_depth must be an integer")
        if value <= 0:
            raise ValueError("tp_depth must be greater than 0")
        self._tp_depth = value

    @property
    def skip_merge(self) -> bool:
        return self._skip_merge

    @skip_merge.setter
    def skip_merge(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("skip_merge must be a boolean")
        self._skip_merge = value

    @property
    def num_neighbors(self) -> int:
        return self._num_neighbors

    @num_neighbors.setter
    def num_neighbors(self, value: int):
        if not isinstance(value, int):
            raise TypeError("num_neighbors must be an integer")
        if value <= 1:
            raise ValueError("num_neighbors must be greater than 1")
        self._num_neighbors = value

    @property
    def avg_sarc_length(self) -> float:
        return self._avg_sarc_length

    @avg_sarc_length.setter
    def avg_sarc_length(self, value: float):
        if not isinstance(value, float):
            raise TypeError("avg_sarc_length must be a float")
        if value <= 0.0:
            raise ValueError("avg_sarc_length must be greater than 0.0")
        self._avg_sarc_length = value

    @property
    def max_sarc_length(self) -> float:
        return self._max_sarc_length

    @max_sarc_length.setter
    def max_sarc_length(self, value: float):
        if not isinstance(value, float):
            raise TypeError("max_sarc_length must be a float")
        if value <= self.avg_sarc_length:
            raise ValueError("max_sarc_length must be greater than "
                             "avg_sarc_length")
        self._max_sarc_length = value

    @property
    def coeff_avg_length(self) -> float:
        return self._coeff_avg_length

    @coeff_avg_length.setter
    def coeff_avg_length(self, value: float):
        if not isinstance(value, float):
            raise TypeError("coeff_avg_length must be a float")
        if value <= 0.0:
            raise ValueError("coeff_avg_length must be greater than 0.0")
        self._coeff_avg_length = value

    @property
    def coeff_neighbor_length(self) -> float:
        return self._coeff_neighbor_length

    @coeff_neighbor_length.setter
    def coeff_neighbor_length(self, value: float):
        if not isinstance(value, float):
            raise TypeError("coeff_neighbor_length must be a float")
        if value <= 0.0:
            raise ValueError("coeff_neighbor_length must be greater than 0.0")
        self._coeff_neighbor_length = value

    @property
    def coeff_neighbor_angle(self) -> float:
        return self._coeff_neighbor_angle

    @coeff_neighbor_angle.setter
    def coeff_neighbor_angle(self, value: float):
        if not isinstance(value, float):
            raise TypeError("coeff_neighbor_angle must be a float")
        if value <= 0.0:
            raise ValueError("coeff_neighbor_angle must be greater than 0.0")
        self._coeff_neighbor_angle = value

    @property
    def score_threshold(self) -> float:
        return self._score_threshold

    @score_threshold.setter
    def score_threshold(self, value: float):
        if not isinstance(value, float):
            raise TypeError("score_threshold must be a float")
        if value <= 0.0:
            raise ValueError("score_threshold must be greater than 0.0")
        self._score_threshold = value

    @property
    def angle_threshold(self) -> float:
        return self._angle_threshold

    @angle_threshold.setter
    def angle_threshold(self, value: float):
        if not isinstance(value, float):
            raise TypeError("angle_threshold must be a float")
        if value <= 1.0:
            raise ValueError("angle_threshold must be greater than 1.0")
        self._angle_threshold = value

    def print(self):
        for attribute in self.__dict__:
            # Strip the leading underscore and get the property's value using its getter.
            public_attribute_name = attribute.lstrip('_')
            value = getattr(self, public_attribute_name)
            print(f"{public_attribute_name} = {value}")
