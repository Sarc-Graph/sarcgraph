from src.sarcgraph import SarcGraph


def test_detected_zdisc_num():
    sg = SarcGraph(output_dir='test-output', input_type='video')
    filtered_data = sg.preprocessing(input_path='samples/sample_0.avi')
    contours = sg.zdisc_detection(filtered_data)
    assert len(contours) == 80


def test_segmentation_frames_num():
    sg = SarcGraph(output_dir='test-output', input_type='video')
    zdiscs_info = sg.zdisc_segmentation(input_path='samples/sample_0.avi')
    assert len(zdiscs_info) == 80
