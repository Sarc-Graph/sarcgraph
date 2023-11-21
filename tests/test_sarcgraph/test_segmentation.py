import pytest
import numpy as np
import pandas as pd
from sarcgraph import SarcGraph
from tempfile import TemporaryDirectory


@pytest.fixture
def sg():
    with TemporaryDirectory() as tmpdirname:
        sg = SarcGraph(output_dir=tmpdirname)
        yield sg


def test_detect_contours_valid_input(sg):
    frames = np.random.rand(5, 100, 100)
    contours = sg._detect_contours(frames)
    assert isinstance(contours, np.ndarray)
    assert contours.ndim == 1
    assert contours.shape[0] == 5


def test_detect_contours_invalid_input(sg):
    invalid_frames = np.random.rand(100, 100)
    with pytest.raises(ValueError):
        sg._detect_contours(invalid_frames)


def test_validate_contours(sg):
    sg.config.zdisc_min_length = 5
    sg.config.zdisc_max_length = 15
    contours = [np.random.rand(np.random.randint(1, 20), 2) for _ in range(10)]
    valid_contours = sg._validate_contours(contours)
    for contour in valid_contours:
        assert 5 <= len(contour) <= 15


def test_find_frame_contours(sg):
    frame = np.random.rand(100, 100)
    contours = sg._find_frame_contours(frame)
    assert isinstance(contours, list)
    for contour in contours:
        assert isinstance(contour, np.ndarray)
        assert contour.ndim == 2


def test_detect_correct_number_of_contours(sg):
    def generate_test_frame(num_contours):
        frame = np.zeros((100, 100))
        step = 25 // num_contours
        for i in range(1, num_contours + 1):
            frame[
                i * step : i * step + 5, i * step : i * step + 5  # noqa: E203
            ] = 255
        return frame

    frames = np.array([generate_test_frame(i) for i in range(1, 5)])
    contours = sg._detect_contours(frames)
    for i, frame_contours in enumerate(contours):
        assert len(frame_contours) == i + 1


def test_process_contours(sg):
    def mock_processing_function(contour):
        return {"length": len(contour)}

    contours_all = [
        [np.random.rand(np.random.randint(5, 15), 2) for _ in range(3)],
        [np.random.rand(np.random.randint(5, 15), 2) for _ in range(2)],
    ]
    processing_functions = [mock_processing_function, sg._zdisc_center]

    result_df = sg._process_contours(contours_all, processing_functions)
    assert isinstance(result_df, pd.DataFrame)
    assert "frame" in result_df.columns
    assert "length" in result_df.columns
    assert "x" in result_df.columns
    assert "y" in result_df.columns


def test_zdisc_center(sg):
    contour = np.array(
        [[0, 1], [1, 0], [2, 1], [2, 2], [1, 3], [0, 2], [0, 1]]
    )
    center = sg._zdisc_center(contour)
    assert center == {"x": 1.0, "y": 1.5}


def test_zdisc_endpoints(sg):
    contour = np.array(
        [[0, 1], [1, 0], [2, 1], [2, 2], [1, 3], [0, 2], [0, 1]]
    )
    endpoints = sg._zdisc_endpoints(contour)
    assert endpoints == {"p1_x": 1, "p1_y": 0, "p2_x": 1, "p2_y": 3}


def test_zdisc_segmentation_with_input_file(sg):
    sg.config.input_type = "image"
    zdiscs = sg.zdisc_segmentation(input_file="samples/sample_5.png")
    expected_num_zdiscs = 81
    assert len(zdiscs) == expected_num_zdiscs


def test_zdisc_segmentation_with_raw_frames(sg):
    sg.config.input_type = "image"
    raw_frames = sg.load_data("samples/sample_horizontal.tif")
    zdiscs = sg.zdisc_segmentation(raw_frames=raw_frames)
    assert isinstance(zdiscs, pd.DataFrame)
    assert len(zdiscs) == 20


def test_zdisc_segmentation_with_filtered_frames(sg):
    sg.config.input_type = "image"
    raw_frames = sg.load_data("samples/sample_vertical.tif")
    filtered_frames = sg.filter_frames(raw_frames)
    zdiscs = sg.zdisc_segmentation(filtered_frames=filtered_frames)
    assert isinstance(zdiscs, pd.DataFrame)
    assert len(zdiscs) == 20


def test_zdisc_segmentation_with_custom_processing(sg):
    def mock_processing_function(contour):
        return {"length": len(contour)}

    zdiscs = sg.zdisc_segmentation(
        input_file="samples/sample_0.avi",
        processing_functions=[mock_processing_function],
        sigma=1.0,
    )
    assert "length" in zdiscs.columns


def test_zdisc_segmentation_error_handling(sg):
    with pytest.raises(ValueError):
        sg.zdisc_segmentation()
