from src.sarcgraph import SarcGraph
from src.sarcgraph_tools import SarcGraphTools


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    sg = SarcGraph()
    sg_tools = SarcGraphTools()

    input_file = "samples/sample_0.avi"
    sg.sarcomere_detection(input_file)
    sg_tools.time_series.sarcomeres_gpr()
    sg_tools.run_all(input_file)
