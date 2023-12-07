import pathlib

import pytest


@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    yield tmp_path
