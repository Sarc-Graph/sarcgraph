import pytest
import numpy as np
import pandas as pd
import os

from src.sarcgraph import SarcGraph

sg_vid = SarcGraph("test", "video")
sg_img = SarcGraph("test", "image")


def test_output_dir_not_string():
    with pytest.raises(TypeError):
        SarcGraph(output_dir=1)


def test_wrong_file_type():
    with pytest.raises(ValueError):
        SarcGraph(file_type="Image")


def test_video_loader_avi():
    frames = sg_vid._to_gray(sg_vid._data_loader("samples/sample_0.avi"))
    assert frames.shape == (80, 368, 368, 1)


def test_video_loader_tif():
    frames = sg_vid._to_gray(sg_vid._data_loader("samples/sample_3.tif"))
    assert frames.shape == (30, 512, 512, 1)


def test_image_loader():
    frame = sg_img._to_gray(sg_img._data_loader("samples/sample_5.png"))
    assert frame.shape == (1, 368, 368, 1)


def test_video_load_as_image():
    with pytest.raises(
        ValueError,
        match=(
            "Failed to load video correctly! Manually load the video into a "
            "numpy array and input to the function as raw_frames."
        ),
    ):
        sg_vid._to_gray(sg_vid._data_loader("samples/sample_4.png"))


def test_image_load_as_video():
    with pytest.raises(
        ValueError,
        match=(
            "Trying to load a video while file_type='image'. Load the image "
            "manually or change the file_type to 'video."
        ),
    ):
        sg_img._to_gray(sg_img._data_loader("samples/sample_3.tif"))


def test_save_numpy_file_name_str():
    with pytest.raises(TypeError):
        SarcGraph()._save_numpy([0], 0)


def test_save_dataframe_file_name_str():
    with pytest.raises(TypeError):
        SarcGraph()._save_dataframe([0], 0)


def test_save_dict():
    test_dict = {"test": [1]}
    with pytest.raises(TypeError):
        sg_img._save_numpy(test_dict, "test-dict")
    with pytest.raises(TypeError):
        sg_img._save_dataframe(test_dict, "test-dict")


def test_save_numpy():
    test_np = np.ones((1, 2))
    test_list = [1, 2]
    file_name_np = "test-np"
    file_name_list = "test-list"
    sg_img._save_numpy(test_np, file_name_np)
    sg_img._save_numpy(test_list, file_name_list)
    assert os.path.exists(f"./{sg_img.output_dir}/{file_name_np}.npy")
    assert os.path.exists(f"./{sg_img.output_dir}/{file_name_list}.npy")


def test_save_dataframe():
    test_df = pd.DataFrame.from_dict({"test": [1]})
    file_name_df = "test-df"
    sg_img._save_dataframe(test_df, file_name_df)
    assert os.path.exists(f"./{sg_img.output_dir}/{file_name_df}.csv")


def test_filtered_data_input_format():
    data_fmt_1 = np.ones((3, 4, 4, 2, 1))
    data_fmt_2 = np.ones((3, 4, 4, 2))
    data_fmt_3 = np.ones((3, 4, 4))
    data_fmt_4 = np.ones((4, 4))
    with pytest.raises(ValueError):
        sg_img._filter_frames(data_fmt_1)
    with pytest.raises(ValueError):
        sg_img._filter_frames(data_fmt_2)
    with pytest.raises(ValueError):
        sg_img._filter_frames(data_fmt_4)
    assert sg_img._filter_frames(data_fmt_3).shape == data_fmt_3.shape


def test_filtered_data_output():
    test_data = np.zeros((2, 65, 65))
    test_data[0, :, 32] = 1
    test_data[1, 32, :] = 1
    filtered_data = sg_img._filter_frames(test_data)
    assert np.array_equal(
        np.argmax(filtered_data[0], axis=1), 32 * np.ones(65)
    )
    assert np.array_equal(
        np.argmax(filtered_data[1], axis=0), 32 * np.ones(65)
    )
