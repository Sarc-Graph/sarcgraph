# import pytest
# import numpy as np
# import pandas as pd
import os

from sarcgraph.sg import SarcGraph
from sarcgraph.sg_tools import SarcGraphTools

sg = SarcGraph()
sg_tools = SarcGraphTools(quality=50, include_eps=True)


def test_zdiscs_and_sarcs():
    sg_tools.visualization.zdiscs_and_sarcs()
    file_name = f"{sg_tools.output_dir}/zdiscs-sarcs-frame-0"
    assert os.path.exists(file_name + ".png")
    assert os.path.exists(file_name + ".eps")


def test_contraction():
    sg_tools.visualization.contraction()
    file_name = f"{sg_tools.output_dir}/contract_anim.gif"
    assert os.path.exists(file_name)
    assert not os.path.exists("tmp")


def test_normalized_sarcs_length():
    sg_tools.visualization.normalized_sarcs_length()
    file_name = f"{sg_tools.output_dir}/normalized_sarcomeres_length"
    assert os.path.exists(file_name + ".png")
    assert os.path.exists(file_name + ".eps")


def test_OOP():
    sg_tools.visualization.OOP()
    file_name = f"{sg_tools.output_dir}/recovered_OOP"
    assert os.path.exists(file_name + ".png")
    assert os.path.exists(file_name + ".eps")


def test_F():
    sg_tools.visualization.F()
    file_name = f"{sg_tools.output_dir}/recovered_F"
    assert os.path.exists(file_name + ".png")
    assert os.path.exists(file_name + ".eps")


def test_J():
    sg_tools.visualization.J()
    file_name = f"{sg_tools.output_dir}/recovered_J"
    assert os.path.exists(file_name + ".png")
    assert os.path.exists(file_name + ".eps")


def test_F_eigenval_animation():
    sg_tools.visualization.F_eigenval_animation()
    assert os.path.exists(f"{sg_tools.output_dir}/F_anim.gif")
    assert not os.path.exists("tmp")


def test_timeseries_params():
    sg_tools.visualization.timeseries_params()
    file_name = f"{sg_tools.output_dir}/histogram_time_constants"
    assert os.path.exists(file_name + ".png")
    assert os.path.exists(file_name + ".eps")


def test_dendrogram():
    dist_fn1 = "dtw"
    dist_fn2 = "euclidean"
    sg_tools.visualization.dendrogram(dist_fn1)
    sg_tools.visualization.dendrogram(dist_fn2)
    file_name_1 = f"{sg_tools.output_dir}/dendrogram_{dist_fn1}.pdf"
    file_name_2 = f"{sg_tools.output_dir}/dendrogram_{dist_fn2}.pdf"
    assert os.path.exists(file_name_1)
    assert os.path.exists(file_name_2)


def test_spatial_graph():
    sg_tools.visualization.spatial_graph()
    file_name = f"{sg_tools.output_dir}/spatial-graph"
    assert os.path.exists(file_name + ".png")
    assert os.path.exists(file_name + ".eps")


def test_tracked_vs_untracked():
    input_file = "samples/sample_0.avi"
    sg_tools.visualization.tracked_vs_untracked(input_file, 0, 10)
    file_name = f"{sg_tools.output_dir}/length-comparison"
    assert os.path.exists(file_name + ".png")
    assert os.path.exists(file_name + ".eps")
    file_name = f"{sg_tools.output_dir}/width-comparison"
    assert os.path.exists(file_name + ".png")
    assert os.path.exists(file_name + ".eps")
    file_name = f"{sg_tools.output_dir}/angle-comparison"
    assert os.path.exists(file_name + ".png")
    assert os.path.exists(file_name + ".eps")
