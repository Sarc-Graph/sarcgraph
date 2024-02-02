import numpy as np
import pandas as pd
import os

from sarcgraph.sg_tools import SarcGraphTools
from sarcgraph.sg import SarcGraph

sg = SarcGraph()
sg_tools = SarcGraphTools()


def test_sampler():
    s1 = np.array([1, 1, 1, 1])
    s2 = np.array([1, 2, 3, 4])
    res1 = sg_tools.analysis._sampler(s1, 1, 2, 5)
    res2 = sg_tools.analysis._sampler(s2, 0, 3, 5)
    assert np.all(res1 == 0)
    assert np.all(res2 < 0)


def test_angular_mean():
    s1 = [np.pi / 6, 5 * np.pi / 6]
    res1 = sg_tools.analysis._angular_mean(s1)
    assert np.isclose(res1[0], np.pi / 2)
    assert np.isclose(res1[1], 0.5)


def test_angular_sampler():
    s1 = np.ones(10)
    res1 = sg_tools.analysis._angular_sampler(s1, 0, 2, 5, 5)
    assert np.allclose(res1[0], -1)
    assert np.allclose(res1[1], 1)


def test_compute_F_J():
    F, J = sg_tools.analysis.compute_F_J(adjust_reference=True)
    assert F is not None
    assert J is not None


def test_compute_OOP():
    OOP, OOP_vec = sg_tools.analysis.compute_OOP()
    assert OOP_vec is not None
    assert OOP is not None


def test_compute_metrics():
    metrics = sg_tools.analysis.compute_metrics()
    metrics_list = set(["C_OOP", "C_iso", "OOP", "s_avg", "s_til"])
    assert set(metrics.keys()) == metrics_list


def test_compute_ts_params():
    ts_params = sg_tools.analysis.compute_ts_params()
    assert isinstance(ts_params, pd.DataFrame)


def test_create_spatial_graph():
    sg_tools.analysis.create_spatial_graph("samples/sample_0.avi")
    assert os.path.exists(f"{sg_tools.output_dir}/spatial-graph.pkl")
