import pytest
from sarcgraph import SarcGraph
from sarcgraph.config import Config
from pathlib import Path
from .mock import MockConfig
from tempfile import TemporaryDirectory


@pytest.fixture
def temp_config():
    with TemporaryDirectory() as tempdirname:
        config = Config()
        config.output_dir = tempdirname
        yield config


def test_init_with_default_config(temp_config):
    sg = SarcGraph(config=temp_config)
    assert isinstance(sg.config, Config)
    assert sg.config.input_type == "video"


def test_init_with_custom_config(temp_config):
    temp_config.input_type = "image"
    sg = SarcGraph(config=temp_config)
    assert sg.config.input_type == "image"


def test_init_with_kwargs(temp_config):
    sg = SarcGraph(config=temp_config, input_type="image")
    assert sg.config.input_type == "image"


def test_init_with_invalid_kwargs(temp_config):
    with pytest.raises(ValueError):
        SarcGraph(config=temp_config, input_type="invalid")
    with pytest.raises(AttributeError):
        SarcGraph(config=temp_config, invalid="invalid")


def test_output_dir_creation(temp_config):
    SarcGraph(config=temp_config)
    assert Path(temp_config.output_dir).exists()


def test_print_config():
    mock_config = MockConfig()
    sg = SarcGraph(config=mock_config)
    sg.print_config()
    assert mock_config.print_called
    mock_config.cleanup()


def test_upate_config_valid_kwargs(temp_config):
    sg = SarcGraph(config=temp_config)
    sg._update_config(input_type="image")
    assert sg.config.input_type == "image"


def test_upate_config_invalid_kwargs(temp_config):
    sg = SarcGraph(config=temp_config)
    with pytest.raises(ValueError):
        sg._update_config(input_type="invalid")
    with pytest.raises(AttributeError) as excinfo:
        sg._update_config(invalid="invalid")
    assert "invalid is not a valid configuration parameter." in str(
        excinfo.value
    )


def test_create_output_dir(temp_config):
    sg = SarcGraph(config=temp_config)
    assert Path(sg.config.output_dir).exists()
