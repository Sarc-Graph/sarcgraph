import numpy as np
import pandas as pd
import os
import pytest

from sarcgraph.sg import SarcGraph

sg_vid = SarcGraph()


def test_zdisc_tracking_output_fmt():
    tracked_zdiscs = sg_vid.zdisc_tracking(
        "samples/sample_0.avi", save_output=False
    )
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
    sg_test = SarcGraph("test")
    _ = sg_test.zdisc_tracking("samples/sample_0.avi")
    assert os.path.exists(f"./{sg_test.output_dir}/tracked-zdiscs.csv")


def test_zdisc_tracking():
    tracked_zdiscs = sg_vid.zdisc_tracking(
        "samples/sample_0.avi", save_output=False
    )
    zdiscs_num_in_frames = tracked_zdiscs.groupby("particle")[
        "particle"
    ].count()
    assert np.array_equal(zdiscs_num_in_frames, 80 * np.ones(81))
    assert len(tracked_zdiscs.particle.unique()) == 81
    segmented_zdiscs = pd.read_csv(
        "tests/test_data/segmented-zdiscs.csv", index_col=[0]
    )
    expected = pd.read_csv(
        "tests/test_data/test-tracked-zdiscs.csv", index_col=[0]
    )
    tracked_zdiscs = sg_vid.zdisc_tracking(
        segmented_zdiscs=segmented_zdiscs, save_output=False
    )
    assert np.allclose(tracked_zdiscs.to_numpy(), expected.to_numpy())


def test_zdisc_tracking_input():
    with pytest.raises(TypeError):
        sg_vid.zdisc_tracking(segmented_zdiscs=np.ones((2, 4)))
    columns = ["one", "two", "three"]
    dataframe = pd.DataFrame(columns=columns)
    with pytest.raises(ValueError):
        sg_vid.zdisc_tracking(segmented_zdiscs=dataframe)
