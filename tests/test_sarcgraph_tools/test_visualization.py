import pytest
import numpy as np
import pandas as pd
import os

from src.sarcgraph import SarcGraph
from src.sarcgraph_tools import SarcGraphTools

sg = SarcGraph()
sg.sarcomere_detection("samples/sample_0.avi")

sg_tools = SarcGraphTools()
sg_tools.time_series.sarcomeres_gpr()


def test_zdiscs_and_sarcs():
    sg_tools.visualization.zdiscs_and_sarcs()
    file_path = f"{sg_tools.output_dir}/zdiscs-sarcs-frame-0.png"
    assert os.path.exists(file_path)


def test_contraction():
    sg_tools.visualization.contraction()
    file_path = f"{sg_tools.output_dir}/contract_anim.gif"
    assert os.path.exists(file_path)
    assert not os.path.exists("tmp")


def test_plot_normalized_sarcs_length():
    sg_tools.visualization.plot_normalized_sarcs_length()
    file_path = f"{sg_tools.output_dir}/normalized_sarcomeres_length.png"
    assert os.path.exists(file_path)


def test_plot_OOP():
    sg_tools.visualization.plot_OOP()
    file_path = f"{sg_tools.output_dir}/recovered_OOP.png"
    assert os.path.exists(file_path)


def test_plot_F():
    sg_tools.visualization.plot_F()
    file_path = f"{sg_tools.output_dir}/recovered_F.png"
    assert os.path.exists(file_path)


def test_plot_J():
    sg_tools.visualization.plot_F()
    file_path = f"{sg_tools.output_dir}/recovered_J.png"
    assert os.path.exists(file_path)


def test_F_eigenval_animation():
    sg_tools.visualization.F_eigenval_animation()
    assert os.path.exists(f"{sg_tools.output_dir}/F_anim.gif")


"""timeseries_params function in the analysis class should be fixed"""


def test_timeseries_params():
    sg_tools.visualization.timeseries_params()
    file_path = f"{sg_tools.output_dir}/histogram_time_constants.png"
    assert os.path.exists(file_path)


def test_dendrogram():
    sg_tools.visualization.dendrogram("dtw")
    file_path = f"{sg_tools.output_dir}/dendrogram_dtw.pdf"
    assert os.path.exists(file_path)


"""sparial_graph function in both analysis and visualization should be checked"""
# def test_spatial_graph():

"""tracked_vs_untracked function needs to be fixed"""
# def test_tracked_vs_untracked():
#     sg_tools.visualization.tracked_vs_untracked(5, 25)
