import pytest
import numpy as np
import pandas as pd
from sarcgraph import SarcGraph
from tempfile import TemporaryDirectory
from pathlib import Path

IMAGE_PATH = "samples/sample_4.png"
TIFF_IMAGE_PATH = "samples/sample_vertical.tif"
VIDEO_PATH = "samples/sample_0.avi"
TIFF_VIDEO_PATH = "samples/sample_6.tif"


@pytest.fixture
def sg():
    with TemporaryDirectory() as tmpdirname:
        sg = SarcGraph(output_dir=tmpdirname)
        yield sg


def test_load_data_valid_image(sg):
    sg.config.input_type = "image"
    data = sg.load_data(IMAGE_PATH)
    assert isinstance(data, np.ndarray) and data.ndim == 2


def test_load_data_valid_tiff_image(sg):
    sg.config.input_type = "image"
    data = sg.load_data(TIFF_IMAGE_PATH)
    assert isinstance(data, np.ndarray) and data.ndim == 2


def test_load_data_valid_video(sg):
    sg.config.input_type = "video"
    data = sg.load_data(VIDEO_PATH)
    assert (
        isinstance(data, np.ndarray) and data.ndim == 3 and data.shape[0] > 1
    )


def test_load_data_valid_tiff_video(sg):
    sg.config.input_type = "video"
    data = sg.load_data(TIFF_VIDEO_PATH)
    assert (
        isinstance(data, np.ndarray) and data.ndim == 3 and data.shape[0] > 1
    )


def test_load_data_no_file_path(sg):
    with pytest.raises(ValueError):
        sg.load_data()


def test_load_data_invalid_file_path(sg):
    with pytest.raises(Exception):
        sg.load_data("invalid_file_path")


def test_check_validity_image(sg):
    data = np.random.rand(100, 100)
    assert sg._check_validity(data, "image")


def test_check_validity_invalid_image(sg):
    data = np.random.rand(3, 100, 100)
    with pytest.raises(ValueError):
        sg._check_validity(data, "image")


def test_check_validity_video(sg):
    data = np.random.rand(10, 100, 100)
    assert sg._check_validity(data, "video")


def test_check_validity_invalid_video(sg):
    data = np.random.rand(100, 100)
    with pytest.raises(ValueError):
        sg._check_validity(data, "video")


def test_save_numpy_array(sg):
    data = np.array([1, 2, 3])
    sg.save_data(data, "test_array")
    assert Path(f"{sg.config.output_dir}/test_array.npy").exists()


def test_save_list(sg):
    data = [1, 2, 3]
    sg.save_data(data, "test_list")
    assert Path(f"{sg.config.output_dir}/test_list.npy").exists()


def test_save_pandas_dataframe(sg):
    data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    sg.save_data(data, "test_dataframe")
    assert Path(f"{sg.config.output_dir}/test_dataframe.csv").exists()


def test_save_invalid_data_type(sg):
    with pytest.raises(TypeError):
        sg.save_data([1, 2, 3], 123)


def test_invalid_data_type(sg):
    with pytest.raises(TypeError):
        sg.save_data("invalid_data_type", "test_file")


def test_filter_frames_single_image(sg):
    image = np.random.rand(100, 100)
    filtered_image = sg.filter_frames(image)
    assert filtered_image.ndim == 3
    assert filtered_image.shape[0] == 1


def test_filter_frames_image_stack(sg):
    stack = np.random.rand(5, 100, 100)
    filtered_stack = sg.filter_frames(stack)
    assert filtered_stack.ndim == 3
    assert filtered_stack.shape[0] == 5


def test_filter_frames_invalid_input(sg):
    invalid_data = np.random.rand(10)
    with pytest.raises(ValueError):
        sg.filter_frames(invalid_data)

    invalid_data = np.random.rand(10, 10, 10, 10)
    with pytest.raises(ValueError):
        sg.filter_frames(invalid_data)
