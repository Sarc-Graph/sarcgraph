import pytest
from tempfile import TemporaryDirectory
from sarcgraph import SarcGraph, SarcGraphTools


@pytest.fixture(scope="session")
def sg_tools():
    with TemporaryDirectory() as tmpdirname:
        sg = SarcGraph(output_dir=tmpdirname)
        sg_tools = SarcGraphTools(
            input_dir=tmpdirname, quality=50, include_eps=True
        )

        file_address = "samples/sample_0.avi"
        frames = sg.load_data(file_address)[0:10]
        _, _ = sg.sarcomere_detection(raw_frames=frames)

        sg_tools.time_series.sarcomeres_gpr()
        sg_tools._run_all(file_address)

        yield sg_tools
