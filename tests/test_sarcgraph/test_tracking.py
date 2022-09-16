import numpy as np
import pandas as pd
from src.sarcgraph import SarcGraph


def test_output_format():
    sg = SarcGraph(output_dir="test-output", input_type="video")
    tracked_zdiscs = sg.zdisc_tracking(input_path="samples/sample_0.avi")
    output_columns = set(tracked_zdiscs.columns)
    assert type(tracked_zdiscs) == pd.DataFrame
    assert set(("frame", "x", "y", "particle")).issubset(output_columns)


def test_load_zdiscs_info():
    sg = SarcGraph(output_dir="test-output", input_type="video")
    sg.zdisc_segmentation(input_path="samples/sample_0.avi", save_data=True)
    zdiscs_info = pd.read_pickle(f"{sg.output_dir}/zdiscs-info.pkl")
    tracked_zdiscs = sg.zdisc_tracking(zdiscs_info=zdiscs_info)
    assert len(tracked_zdiscs)


def test_tracking_results():
    sg = SarcGraph(output_dir="test-output", input_type="video")
    tracked_zdiscs = sg.zdisc_tracking(input_path="samples/sample_0.avi")
    zdiscs_in_num_frame = tracked_zdiscs.groupby("particle")["particle"].count()
    assert np.array_equal(zdiscs_in_num_frame, 80 * np.ones(81))
    assert len(tracked_zdiscs.particle.unique()) == 81
