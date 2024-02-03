import pytest
import numpy as np

from sarcgraph import SarcGraphTools

def test_sg_tools_input_dir():
    with pytest.raises(FileNotFoundError):
        SarcGraphTools("worng-input")


def test_dtw_distance(sg_tools):
    s1 = np.array([1, 2, 3])
    s2 = np.array([2, 2, 2, 3, 4])
    res = sg_tools.time_series._dtw_distance(s1, s2)
    assert np.isclose(res**2, 2)


def test_dtw_distance_error(sg_tools):
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
        sg_tools.time_series._dtw_distance(s1_np, s2_2d)
    with pytest.raises(ValueError):
        sg_tools.time_series._dtw_distance(s1_2d, s2_np)


def test_gpr(sg_tools):
    s = np.array([0, 1, np.nan, 3, 4])
    res = sg_tools.time_series._gpr(s)
    res_expected = [0, 1, 2, 3, 4]
    assert np.allclose(res, res_expected, atol=0.01)


def test_sarcomeres_gpr(sg_tools):
    sg_tools = SarcGraphTools(
        input_dir="./tests/test_data", save_results=False
    )
    sarcomeres_gpr = sg_tools.time_series.sarcomeres_gpr()

    assert sarcomeres_gpr.isna().sum().sum() == 0
    assert set(sarcomeres_gpr.frame.unique()) == set([0, 1, 2, 3, 4])
    assert "length_norm" in sarcomeres_gpr
