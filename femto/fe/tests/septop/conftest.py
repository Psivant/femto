import pathlib

import pytest

from femto.fe.tests.systems import CDK2_SYSTEM, create_cdk2_input_directory


@pytest.fixture
def mock_bfe_directory(tmp_cwd) -> pathlib.Path:
    root_dir = tmp_cwd / "inputs"

    create_cdk2_input_directory(root_dir)
    (root_dir / "Morph.in").write_text(
        f"{CDK2_SYSTEM.ligand_1_name}~{CDK2_SYSTEM.ligand_1_name}\n"
        f"{CDK2_SYSTEM.ligand_1_name}~{CDK2_SYSTEM.ligand_2_name}"
    )

    return root_dir
