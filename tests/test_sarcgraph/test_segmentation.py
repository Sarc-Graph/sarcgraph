import pytest

from sarcgraph.sarcgraph import SarcGraph

def test_detected_zdisc_num():
    sg = SarcGraph(output_dir='test-output', input_type='video')
    filtered_data = sg.preprocessing(input_path='samples/sample_0.avi')
    contours = sg.zdisc_detection(filtered_data)
    assert len(contours) == 81