import pytest
import numpy as np
import networkx as nx
import pandas as pd
from sarcgraph import SarcGraph
from tempfile import TemporaryDirectory


@pytest.fixture
def sg():
    with TemporaryDirectory() as tmpdirname:
        sg = SarcGraph(output_dir=tmpdirname)
        yield sg


def test_zdisc_to_graph(sg):
    zdiscs = np.array(
        [[0, 0, 1], [1, 0, 2], [0, 1, 3], [1, 1, 4], [0.25, 0.25, 5]]
    )
    G = sg._zdisc_to_graph(zdiscs)
    assert isinstance(G, nx.Graph)
    assert len(G.nodes) == 5
    assert len(G.edges) == 8


def test_graph_initialization(sg):
    zdiscs = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 3]])
    G = sg._graph_initialization(zdiscs)
    assert isinstance(G, nx.Graph)
    assert all(
        isinstance(G.nodes[node]["pos"], np.ndarray) for node in G.nodes
    )
    assert all("particle_id" in G.nodes[node] for node in G.nodes)


def test_find_nearest_neighbors(sg):
    sg.config.num_neighbors = 1
    zdiscs = np.array([[0, 0], [1, 1], [2, 2]])
    nearest_neighbors = sg._find_nearest_neighbors(zdiscs)
    assert nearest_neighbors.shape == (
        len(zdiscs),
        sg.config.num_neighbors + 1,
    )


def test_add_edges(sg):
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    nearest_neighbors = np.array([[0, 1], [1, 2], [2, 0]])
    G = sg._add_edges(G, nearest_neighbors)
    assert G.number_of_edges() == len(nearest_neighbors)


def test_sarc_vector(sg):
    G = nx.Graph()
    G.add_node(0, pos=np.array([0, 0]))
    G.add_node(1, pos=np.array([1, 1]))
    vector, length = sg._sarc_vector(G, 0, 1)
    expected_vector = np.array([1, 1])
    expected_length = np.sqrt(2)
    assert np.array_equal(vector, expected_vector)
    assert np.isclose(length, expected_length)


def test_length_score(sg):
    length_score = sg._length_score(10, 12)
    expected_score = 1 / (1 + 0.2)
    assert np.isclose(length_score, expected_score)


def test_sarcs_angle_and_angle_score(sg):
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    angle = sg._sarcs_angle(v1, v2, 1, 1)
    angle_score = sg._angle_score(v1, v2, 1, 1)
    assert np.isclose(angle, 1)
    assert np.isclose(angle_score, 0)


def test_sarc_score(sg):
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    l1 = l2 = 1
    sg.config.coeff_neighbor_length = 0.5
    sg.config.coeff_neighbor_angle = 0.5
    sarc_score = sg._sarc_score(v1, v2, l1, l2)
    expected_score = 0.5
    assert np.isclose(sarc_score, expected_score)


def test_score_graph(sg):
    zdiscs = np.array([[0, 0, 1], [1, 0, 2], [2, 0, 3], [3, 0, 4]])
    G = sg._zdisc_to_graph(zdiscs)
    sg.config.avg_sarc_length = 1.0
    scored_G = sg._score_graph(G)
    assert scored_G[0][1]["score"] == 3
    assert scored_G[1][2]["score"] == 3
    assert scored_G[2][3]["score"] == 3


def test_prune_graph(sg):
    zdiscs = np.array([[0, 0, 1], [1, 0, 2], [2, 0, 3], [3, 0, 4]])
    G = sg._zdisc_to_graph(zdiscs)
    scored_G = sg._score_graph(G)
    pruned_G = sg._prune_graph(scored_G)
    sg.config.score_threshold = 1.0
    sg.config.angle_threshold = 1.0
    assert np.array_equal(pruned_G.edges, [[0, 1], [1, 2], [2, 3]])


def test_get_connected_zdiscs(sg):
    G = nx.Graph()
    G.add_node(0, particle_id=1)
    G.add_node(1, particle_id=2)
    G.add_edge(0, 1)

    tracked_zdiscs = pd.DataFrame(
        {"frame": [0, 1, 2], "particle": [1, 2, 1], "x": [0, 1, 0]}
    )

    z1, z2 = sg._get_connected_zdiscs(G, tracked_zdiscs, (0, 1))

    assert z1.columns.tolist() == ["frame", "particle_p1", "x_p1"]
    assert z1.values.tolist() == [[0, 1, 0], [2, 1, 0]]
    assert z2.columns.tolist() == ["frame", "particle_p2", "x_p2"]
    assert z2.values.tolist() == [[1, 2, 1]]


def test_initialize_sarc(sg):
    z0 = pd.DataFrame({"frame": [0, 1, 2]})
    z1 = pd.DataFrame({"frame": [0, 1], "x_p1": [0, 1]})
    z2 = pd.DataFrame({"frame": [1, 2], "x_p2": [1, 2]})

    sarc = sg._initialize_sarc(z0, z1, z2)
    assert sarc.columns.tolist() == ["frame", "x_p1", "x_p2"]
    assert len(sarc) == len(z0)
    assert sarc.dropna().values.tolist() == [[1, 1, 1]]
    assert len(sarc.dropna()) == 1


def test_process_sarc(sg):
    sarc = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "sarc_id": [1, 1, 1],
            "zdiscs": ["1, 2", "1, 2", "1, 2"],
            "x_p1": [0, 1, 2],
            "y_p1": [0, 1, 2],
            "x_p2": [1, 2, 3],
            "y_p2": [1, 2, 3],
            "p1_x_p1": [0, 1, 2],
            "p1_y_p1": [0, 1, 2],
            "p2_x_p1": [1, 2, 3],
            "p2_y_p1": [1, 2, 3],
            "p1_x_p2": [1, 2, 3],
            "p1_y_p2": [1, 2, 3],
            "p2_x_p2": [2, 3, 4],
            "p2_y_p2": [2, 3, 4],
        }
    )

    processed_sarc = sg._process_sarc(sarc)

    assert len(processed_sarc) == 3
    assert processed_sarc.columns.tolist() == [
        "frame",
        "sarc_id",
        "x",
        "y",
        "length",
        "width",
        "angle",
        "zdiscs",
    ]
    assert processed_sarc.length.values.tolist() == [np.sqrt(2)] * 3
    assert processed_sarc.width.values.tolist() == [np.sqrt(2)] * 3
    assert processed_sarc.angle.values.tolist() == [3 * np.pi / 4] * 3


def test_process_sarcomeres(sg):
    # Create a mock graph and tracked zdiscs DataFrame
    G = nx.Graph()
    G.add_edge(0, 1)
    G.nodes[0]["particle_id"] = 1
    G.nodes[1]["particle_id"] = 2

    tracked_zdiscs = pd.DataFrame(
        {
            "frame": [0, 0, 1],
            "particle": [1, 2, 1],
            "x": [0, 1, 2],
            "y": [0, 1, 2],
            "p1_x": [0, 1, 2],
            "p1_y": [-0.5, 0.5, 1.5],
            "p2_x": [0, 1, 2],
            "p2_y": [0.5, 1.5, 2.5],
        }
    )

    processed_sarcs = sg._process_sarcomeres(G, tracked_zdiscs)
    assert len(processed_sarcs) == 2
    assert processed_sarcs.columns.tolist() == [
        "frame",
        "sarc_id",
        "x",
        "y",
        "length",
        "width",
        "angle",
        "zdiscs",
    ]
    assert processed_sarcs.dropna().length.values.tolist() == [np.sqrt(2)]
    assert processed_sarcs.dropna().width.values.tolist() == [1]
    assert processed_sarcs.dropna().angle.values.tolist() == [3 * np.pi / 4]
    assert processed_sarcs.zdiscs.values.tolist() == ["1,2", "1,2"]


def test_sarcomere_detection_accuracy(sg):
    sarcs, _ = sg.sarcomere_detection(input_file="./samples/sample_0.avi")
    sarcs_grouped = sarcs.groupby("sarc_id")
    mean_len = sarcs_grouped.length.transform("mean")
    sarcs["length_norm"] = (sarcs.length - mean_len) / mean_len
    sarcs_len_mean = sarcs.groupby("frame").length_norm.mean()
    sarcs_len_mean_gt = np.loadtxt("tests/test_data/gt_sarcs_sample_0.txt")
    error = np.mean((sarcs_len_mean - sarcs_len_mean_gt) ** 2)

    assert error < 1e-4


def test_sarcomere_detection_image(sg):
    sg.config.input_type = "image"

    sarcs, _ = sg.sarcomere_detection(input_file="samples/sample_vertical.tif")
    assert len(sarcs.dropna()) == 19
    assert all(np.isclose(sarcs.angle, np.pi / 2, atol=0.01))

    sarcs, _ = sg.sarcomere_detection(
        input_file="samples/sample_horizontal.tif"
    )
    assert len(sarcs.dropna()) == 19
    assert all(
        np.isclose(sarcs.angle, 0, atol=0.01)
        | np.isclose(sarcs.angle, np.pi, atol=0.01)
    )
