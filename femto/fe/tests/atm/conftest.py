import pathlib

import pytest

from femto.fe.tests.systems import create_cdk2_input_directory


@pytest.fixture
def mock_bfe_directory(tmp_cwd) -> pathlib.Path:
    root_dir = tmp_cwd / "inputs"

    create_cdk2_input_directory(root_dir)
    (root_dir / "Morph.in").write_text("1h1q~1h1q\n1h1q~1oiu")

    return root_dir
