from src.sarcgraph import SarcGraph
import numpy as np


def test_preprocessing():
    sg = SarcGraph(output_dir="test-output", input_type="video")
    processed_data_1 = sg.process_input(input_path="samples/sample_0.avi")
    sg = SarcGraph(output_dir="test-output", input_type="image")
    processed_data_2 = sg.process_input(input_file=np.zeros((58, 28)))
    sg = SarcGraph(output_dir="test-output", input_type="video")
    processed_data_3 = sg.process_input(input_file=np.zeros((4, 58, 28)))
    assert processed_data_1.shape == (80, 368, 368)
    assert processed_data_2.shape == (1, 58, 28)
    assert processed_data_3.shape == (4, 58, 28)


def test_zdisc_contour_detection():
    sg = SarcGraph(output_dir="test-output", input_type="video")
    filtered_data = sg.process_input(input_path="samples/sample_0.avi")
    contours = sg.detect_contours(filtered_data)
    assert len(contours) == 80
    assert len(contours[0]) == 81
    assert contours[0][0].shape[-1] == 2


def test_contour_processor():
    sg = SarcGraph(output_dir="test-output", input_type="video")
    test_contour_1 = [[-1, -1], [1, 1]]
    test_contour_2 = [[-2, -1], [0, -1], [2, -1], [2, 1], [0, 1], [-2, 1]]
    processed_contour_1 = sg.process_contour(test_contour_1)
    processed_contour_2 = sg.process_contour(test_contour_2)
    assert np.array_equal(processed_contour_1, [0, 0, -1, -1, 1, 1])
    assert np.array_equal(processed_contour_2, [0, 0, -2, -1, 2, 1])


def test_zdisc_segmentation():
    sg = SarcGraph(output_dir="test-output", input_type="video")
    zdiscs = sg.zdisc_segmentation(input_path="samples/sample_0.avi")
    assert np.array_equal(
        zdiscs.columns, ["frame", "x", "y", "p1_x", "p1_y", "p2_x", "p2_y"]
    )
    assert len(zdiscs.frame.unique()) == 80
    assert len(zdiscs.loc[zdiscs.frame == 0]) == 81
