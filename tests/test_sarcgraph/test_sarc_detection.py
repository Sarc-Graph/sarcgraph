import numpy as np
from src.sarcgraph import SarcGraph

sg_vid = SarcGraph("test", "video")
sg_img = SarcGraph("test", "image")


def test_sarcomere_detection_accuracy():
    sarcomeres, _ = sg_vid.sarcomere_detection("./samples/sample_0.avi")
    groupby_sarcs = sarcomeres.groupby("sarc_id")
    mean_len = groupby_sarcs.length.transform("mean")
    sarcomeres["length_norm"] = (sarcomeres.length - mean_len) / mean_len
    sarcs_len_mean = sarcomeres.groupby("frame").length_norm.mean()
    sarcs_len_mean_gt = np.loadtxt("tests/test_data/gt_sarcs_sample_0.txt")
    error = np.mean((sarcs_len_mean - sarcs_len_mean_gt) ** 2)

    assert error < 1e-4


def test_sarcomere_detection_image():
    sarcomeres, _ = sg_img.sarcomere_detection("samples/sample_5.png")
    assert not sarcomeres.empty


def test_zdisc_to_graph():
    test_data = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 3], [1, 1, 4]])
    G = sg_vid._zdisc_to_graph(test_data)
    assert len(G.nodes) == 4
    assert len(G.edges) == 6
    assert np.array_equal(list(G.nodes[0].keys()), ["pos", "particle_id"])


def test_score_graph():
    test_data = np.array([[0, 0, 1], [1, 0, 2], [2, 0, 3], [3, 0, 4]])
    G = sg_vid._zdisc_to_graph(test_data)
    G = sg_vid._score_graph(G, 1, 1, 1, 1)
    assert G[0][1]["score"] == 3
    assert G[1][2]["score"] == 3
    assert G[2][3]["score"] == 3


def test_prune_graph():
    test_data = np.array([[0, 0, 1], [1, 0, 2], [2, 0, 3], [3, 0, 4]])
    G = sg_vid._zdisc_to_graph(test_data)
    G = sg_vid._score_graph(G, 1, 1, 1, 1)
    G = sg_vid._prune_graph(G, score_threshold=1, angle_threshold=1)
    assert np.array_equal(G.edges, [[0, 1], [1, 2], [2, 3]])
