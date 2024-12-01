import pathlib

import pytest


@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture
def test_data_dir():
    return pathlib.Path(__file__).parent / "data"
