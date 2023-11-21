import pytest
import numpy as np
import pandas as pd
from sarcgraph import SarcGraph
from tempfile import TemporaryDirectory
from pathlib import Path


@pytest.fixture
def sg():
    with TemporaryDirectory() as tmpdirname:
        sg = SarcGraph(output_dir=tmpdirname)
        yield sg


def test_zdisc_tracking_output_fmt_and_save(sg):
    tracked_zdiscs = sg.zdisc_tracking(input_file="samples/sample_0.avi")
    assert isinstance(tracked_zdiscs, pd.DataFrame)
    assert len(tracked_zdiscs.particle.unique()) == 81
    assert Path(f"{sg.config.output_dir}/tracked_zdiscs.csv").exists()

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
    assert set(output_cols) == set(expected_cols)

    zdiscs_num_in_frames = tracked_zdiscs.groupby("particle")[
        "particle"
    ].count()
    segmented_zdiscs = pd.read_csv(
        "tests/test_data/segmented-zdiscs.csv", index_col=[0]
    )
    expected = pd.read_csv(
        "tests/test_data/test-tracked-zdiscs.csv", index_col=[0]
    )
    tracked_zdiscs = sg.zdisc_tracking(
        segmented_zdiscs=segmented_zdiscs, save_output=False
    )
    assert np.array_equal(zdiscs_num_in_frames, 80 * np.ones(81))
    assert np.allclose(tracked_zdiscs.to_numpy(), expected.to_numpy())


def test_zdisc_tracking_input(sg):
    with pytest.raises(ValueError):
        sg.zdisc_tracking(segmented_zdiscs=np.ones((2, 4)))
    columns = ["one", "two", "three"]
    dataframe = pd.DataFrame(columns=columns)
    with pytest.raises(ValueError):
        sg.zdisc_tracking(segmented_zdiscs=dataframe)
