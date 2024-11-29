import pathlib

import pytest


@pytest.fixture
def test_data_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"
