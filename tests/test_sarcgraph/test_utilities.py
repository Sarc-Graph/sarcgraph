import pytest
import numpy as np

from sarcgraph.sarcgraph import SarcGraph

def test_output_dir_not_specified():
    with pytest.raises(Exception):
        sg = SarcGraph()

def test_video_loaded_as_image():
    sg = SarcGraph('test', 'video')
    assert sg._to_gray(sg.data_loader('samples/sample_3.tif')).shape[0] > 1

def test_filtered_data_input_check():
    sg = SarcGraph('test')
    data_1 = np.ones((3,4,4,2))
    data_2 = np.ones((4,4,1))
    with pytest.raises(ValueError):
        sg.filter_data(data_1)
    with pytest.raises(ValueError):
        sg.filter_data(data_2)

def test_filtered_data_output_check():
    sg = SarcGraph('test')
    data = sg.filter_data(sg._to_gray(sg.data_loader('samples/sample_1.avi')))
    assert len(data.shape) == 3