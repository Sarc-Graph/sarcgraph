from src.sarcgraph import SarcGraph
import numpy as np


def test_sarc_detection_accuracy():
    sg = SarcGraph(output_dir="test-output", input_type="video")
    _, sarcs_info = sg.sarcomere_detection(input_path="samples/sample_0.avi")
    sarcs_length_avg = np.mean(sarcs_info[2, :, :], axis=1, keepdims=True)
    sarcs_length_norm = np.mean(
        (sarcs_info[2, :, :] - sarcs_length_avg) / sarcs_length_avg, axis=0
    )
    sarcs_length_norm_gt = np.loadtxt("tests/test_data/gt_sarcs_sample_0.txt")[0]
    error = np.mean((sarcs_length_norm - sarcs_length_norm_gt) ** 2)

    assert error < 1e-4


def test_sarc_detection_on_image():
    sg = SarcGraph(output_dir="test-output", input_type="image")
    _, sarcs_info = sg.sarcomere_detection(input_path="samples/sample_5.png")
    assert len(sarcs_info)


def test_graph_creation():
    sg = SarcGraph(output_dir="test-output", input_type="image")
    test_clusters = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 3], [1, 1, 4]])
    G = sg.zdisc_clusters_to_graph(test_clusters)
    assert len(G.nodes) == 4
    assert len(G.edges) == 6
    assert np.array_equal(list(G.nodes[0].keys()), ["pos", "particle_id"])


def test_connections_scoring():
    sg = SarcGraph(output_dir="test-output", input_type="image")
    test_clusters = np.array([[0, 0, 1], [1, 0, 2], [2, 0, 3], [3, 0, 4]])
    G = sg.zdisc_clusters_to_graph(test_clusters)
    G = sg.score_connections(G, 1, 1, 1, 1)
    assert G[0][1]["score"] == 3
    assert G[1][2]["score"] == 3
    assert G[2][3]["score"] == 3


def test_valid_connections():
    sg = SarcGraph(output_dir="test-output", input_type="image")
    test_clusters = np.array([[0, 0, 1], [1, 0, 2], [2, 0, 3], [3, 0, 4]])
    G = sg.zdisc_clusters_to_graph(test_clusters)
    G = sg.score_connections(G, 1, 1, 1, 1)
    G = sg.find_valid_connections(G, score_threshold=1, angle_threshold=1)
    assert np.array_equal(G.edges, [[0, 1], [1, 2], [2, 3]])
