import pytest
import numpy as np
import pandas as pd
import os
from src.sarcgraph import SarcGraph

sg_vid = SarcGraph("test", "video")
sg_img = SarcGraph("test", "image")


def test_process_input():
    filtered_frames_1 = sg_vid._process_input("samples/sample_0.avi")
    filtered_frames_2 = sg_vid._process_input(raw_frames=np.zeros((4, 58, 28)))
    filtered_frames_3 = sg_img._process_input(raw_frames=np.zeros((58, 28)))
    assert filtered_frames_1.shape == (80, 368, 368)
    assert filtered_frames_2.shape == (4, 58, 28)
    assert filtered_frames_3.shape == (1, 58, 28)
    assert os.path.exists(f"./{sg_vid.output_dir}/raw-frames.npy")
    assert os.path.exists(f"./{sg_vid.output_dir}/filtered-frames.npy")


def test_detect_contours_input_fmt():
    with pytest.raises(ValueError):
        SarcGraph()._detect_contours(np.ones((4, 4)))
    with pytest.raises(ValueError):
        SarcGraph()._detect_contours(np.ones((1, 4, 4)), min_length="1.0")


def test_detect_contours():
    filtered_frames = sg_vid._process_input("samples/sample_0.avi")
    contours = sg_vid._detect_contours(filtered_frames)
    assert len(contours) == 80
    assert len(contours[0]) == 81
    assert os.path.exists(f"./{sg_vid.output_dir}/contours.npy")


def test_contour_processor():
    test_contour_1 = [[-1, -1], [1, 1]]
    test_contour_2 = [[-2, -1], [0, -1], [2, -1], [2, 1], [0, 1], [-2, 1]]
    with pytest.raises(ValueError):
        sg_vid._process_contour(test_contour_1)
    contour_2 = sg_vid._process_contour(test_contour_2)
    assert np.array_equal(contour_2, [0, 0, -2, -1, 2, 1])


def test_zdiscs_to_pandas():
    with pytest.raises(ValueError):
        sg_vid._zdiscs_to_pandas([np.ones((1, 6))])
    with pytest.raises(ValueError):
        sg_img._zdiscs_to_pandas([np.ones((1, 6)), np.ones((2, 6))])
    with pytest.raises(ValueError):
        sg_img._zdiscs_to_pandas([np.ones((6))])
    with pytest.raises(TypeError):
        sg_img._zdiscs_to_pandas([[1, 1, 1, 1, 1, 1]])
    with pytest.raises(ValueError):
        sg_img._zdiscs_to_pandas([np.ones((4, 5))])
    zdiscs_df = sg_img._zdiscs_to_pandas([np.ones((1, 6))])
    assert isinstance(zdiscs_df, pd.DataFrame)
    columns = ["frame", "x", "y", "p1_x", "p1_y", "p2_x", "p2_y"]
    assert set(zdiscs_df.columns) == set(columns)


def test_zdisc_segmentation():
    zdiscs = sg_vid.zdisc_segmentation("samples/sample_0.avi")
    assert zdiscs.frame.max() == 79
    assert len(zdiscs[zdiscs.frame == 0]) == 81
