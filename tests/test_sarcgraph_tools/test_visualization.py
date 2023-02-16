# import pytest
# import numpy as np
# import pandas as pd
import os

from sarcgraph.sarcgraph import SarcGraph
from sarcgraph.sarcgraph_tools import SarcGraphTools

sg = SarcGraph()
sg_tools = SarcGraphTools(quality=50)


def test_zdiscs_and_sarcs():
    sg_tools.visualization.zdiscs_and_sarcs()
    file_path = f"{sg_tools.output_dir}/zdiscs-sarcs-frame-0.png"
    assert os.path.exists(file_path)


def test_contraction():
    sg_tools.visualization.contraction()
    file_path = f"{sg_tools.output_dir}/contract_anim.gif"
    assert os.path.exists(file_path)
    assert not os.path.exists("tmp")


def test_normalized_sarcs_length():
    sg_tools.visualization.normalized_sarcs_length()
    file_path = f"{sg_tools.output_dir}/normalized_sarcomeres_length.png"
    assert os.path.exists(file_path)


def test_OOP():
    sg_tools.visualization.OOP()
    file_path = f"{sg_tools.output_dir}/recovered_OOP.png"
    assert os.path.exists(file_path)


def test_F():
    sg_tools.visualization.F()
    file_path = f"{sg_tools.output_dir}/recovered_F.png"
    assert os.path.exists(file_path)


def test_J():
    sg_tools.visualization.J()
    file_path = f"{sg_tools.output_dir}/recovered_J.png"
    assert os.path.exists(file_path)


def test_F_eigenval_animation():
    sg_tools.visualization.F_eigenval_animation()
    assert os.path.exists(f"{sg_tools.output_dir}/F_anim.gif")


def test_timeseries_params():
    sg_tools.visualization.timeseries_params()
    file_path = f"{sg_tools.output_dir}/histogram_time_constants.png"
    assert os.path.exists(file_path)


def test_dendrogram():
    sg_tools.visualization.dendrogram("dtw")
    file_path = f"{sg_tools.output_dir}/dendrogram_dtw.pdf"
    assert os.path.exists(file_path)
