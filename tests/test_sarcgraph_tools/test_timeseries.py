import pytest
import numpy as np
import pandas as pd

from src.sarcgraph_tools import SarcGraphTools

sg_tools = SarcGraphTools()


def test_dtw_distance():
    s1 = np.array([1, 2, 3])
    s2 = np.array([2, 2, 2, 3, 4])
    res = sg_tools.time_series._dtw_distance(s1, s2)
    assert np.isclose(res**2, 2)


def test_dtw_distance_error():
    s1 = [1, 2, 3]
    s1_np = np.array(s1)
    s1_2d = s1_np.reshape(-1, 1)
    s2 = [2, 2, 2, 3, 4]
    s2_np = np.array(s2)
    s2_2d = s2_np.reshape(-1, 1)
    with pytest.raises(TypeError):
        sg_tools.time_series._dtw_distance(s1, s2_np)
    with pytest.raises(TypeError):
        sg_tools.time_series._dtw_distance(s1_np, s2)
    with pytest.raises(ValueError):
        sg_tools.time_series._dtw_distance(s1, s2_2d)
    with pytest.raises(ValueError):
        sg_tools.time_series._dtw_distance(s1_2d, s2)


def test_gpr():
    s = np.array([0, 1, np.nan, 3, 4])
    res = sg_tools.time_series._gpr(s)
    res_expected = [0, 1, 2, 3, 4]
    assert np.allclose(res, res_expected, atol=0.01)


def tests_sarcomeres_length_normalize():
    # add tests for this
    assert 0 == 1


def test_sarcomeres_gpr_error():
    with pytest.raises(FileNotFoundError):
        sg_tools.time_series.sarcomeres_gpr()


def test_sarcomeres_gpr():
    sg_tools = SarcGraphTools(input_dir="./tests/test_data")
    sarcomeres_gpr = sg_tools.time_series.sarcomeres_gpr()
    sarcomeres_gpr_expected = pd.read_csv(
        f"{sg_tools.input_dir}/sarcomeres-gpr.csv", index_col=[0]
    )
    assert sarcomeres_gpr.equals(sarcomeres_gpr_expected)
