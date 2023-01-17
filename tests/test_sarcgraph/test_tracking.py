import pytest
import numpy as np
import pandas as pd
import os
from src.sarcgraph import SarcGraph

sg_vid = SarcGraph("test", "video")
sg_img = SarcGraph("test", "image")


def test_zdisc_tracking_output_fmt():
    tracked_zdiscs = sg_vid.zdisc_tracking("samples/sample_0.avi")
    expected_cols = [
        "frame",
        "x",
        "y",
        "p1_x",
        "p1_y",
        "p2_x",
        "p2_y",
        "particle",
        "freq",
    ]
    output_cols = set(tracked_zdiscs.columns)
    assert isinstance(tracked_zdiscs, pd.DataFrame)
    assert set(output_cols) == set(expected_cols)


def test_zdisc_tracking_save():
    _ = sg_vid.zdisc_tracking("samples/sample_0.avi")
    assert os.path.exists(f"./{sg_vid.output_dir}/tracked-zdiscs.csv")


def test_zdisc_tracking():
    tracked_zdiscs = sg_vid.zdisc_tracking("samples/sample_0.avi")
    zdiscs_num_in_frames = tracked_zdiscs.groupby("particle")[
        "particle"
    ].count()
    assert np.array_equal(zdiscs_num_in_frames, 80 * np.ones(81))
    assert len(tracked_zdiscs.particle.unique()) == 81
    sg_vid_new = SarcGraph("tests/test_data")
    tracked_zdiscs = sg_vid_new.zdisc_tracking(save_output=False)
    expected_tracked_zdiscs = pd.read_csv(
        f"{sg_vid_new.output_dir}/test-tracked-zdiscs.csv", index_col=[0]
    )
    assert tracked_zdiscs.equals(expected_tracked_zdiscs)
