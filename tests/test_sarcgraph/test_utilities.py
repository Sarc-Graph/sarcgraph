import pytest
import numpy as np
import pandas as pd

from src.sarcgraph import SarcGraph


def test_output_dir_not_specified():
    with pytest.raises(ValueError):
        SarcGraph()


def test_data_loader():
    sg = SarcGraph("test", "video")
    with pytest.raises(ValueError):
        sg._to_gray(sg.data_loader("samples/sample_4.png"))
    tif_sample_frames = sg._to_gray(sg.data_loader("samples/sample_3.tif")).shape[0]
    avi_sample_frames = sg._to_gray(sg.data_loader("samples/sample_0.avi")).shape[0]

    sg = SarcGraph("test", "image")
    png_sample_frames = sg._to_gray(sg.data_loader("samples/sample_4.png")).shape[0]

    assert tif_sample_frames == 30
    assert avi_sample_frames == 80
    assert png_sample_frames == 1


def test_save_data():
    sg = SarcGraph("test-output")

    list_of_lists = [[1, 2], [2, 3, 4]]
    dict = {"test1": [1], "test2": [2]}
    dataframe = pd.DataFrame.from_dict(dict)
    sg.save_data(list_of_lists, "test")
    sg.save_data(dataframe, "test")

    with pytest.raises(ValueError):
        sg.save_data(1, "test")
    assert type(np.load("test-output/test.npy", allow_pickle=True)) == np.ndarray
    assert type(pd.read_pickle("test-output/test.pkl")) == pd.DataFrame


def test_filtered_data_input():
    sg = SarcGraph("test")
    data_1 = np.ones((3, 4, 4, 2))
    data_2 = np.ones((4, 4, 1))
    with pytest.raises(ValueError):
        sg.filter_frames(data_1)
    with pytest.raises(ValueError):
        sg.filter_frames(data_2)


def test_filtered_data_output():
    test_data = np.zeros((2, 65, 65, 1))
    test_data[0, :, 32] = 1
    test_data[1, 32, :] = 1
    sg = SarcGraph("test")
    filtered_data = sg.filter_frames(sg._to_gray(test_data))
    assert np.array_equal(np.argmax(filtered_data[0], axis=1), 32 * np.ones(65))
    assert np.array_equal(np.argmax(filtered_data[1], axis=0), 32 * np.ones(65))

    sg = SarcGraph("test")
    filtered_data = sg.filter_frames(
        sg._to_gray(sg.data_loader("samples/sample_1.avi"))
    )
    assert len(filtered_data.shape) == 3


def test_zdiscs_info_to_pandas():
    sg = SarcGraph("test")
    with pytest.raises(TypeError):
        sg.zdiscs_info_to_pandas([[1], [2, 3]])
    with pytest.raises(ValueError):
        sg.zdiscs_info_to_pandas([np.array([1])])
    with pytest.raises(ValueError):
        sg.zdiscs_info_to_pandas([np.array([1]), np.array([2, 3])])
