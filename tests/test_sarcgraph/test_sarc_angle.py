import numpy as np

from sarcgraph.sg import SarcGraph

sg_img = SarcGraph("test", "image")


def test_sarcomere_detection_accuracy():
    frame_ver = sg_img._data_loader("samples/sample_vertical.tif")
    frame_hor = sg_img._data_loader("samples/sample_horizontal.tif")

    sarcs_ver, _ = sg_img.sarcomere_detection(raw_frames=frame_ver)
    sarcs_hor, _ = sg_img.sarcomere_detection(raw_frames=frame_hor)

    close_to_90 = np.isclose(sarcs_ver.angle, np.pi / 2, atol=0.01)
    close_to_0_or_180 = np.isclose(sarcs_hor.angle, 0, atol=0.01) | np.isclose(
        sarcs_hor.angle, np.pi, atol=0.01
    )

    assert np.all(close_to_90)
    assert np.all(close_to_0_or_180)
