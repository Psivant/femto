import pathlib

import pydantic
import pytest

from femto.fe.septop._config import (
    SepTopConfig,
    SepTopSetupStage,
    SepTopStates,
    load_config,
)


@pytest.fixture
def mock_config_path(tmp_cwd) -> pathlib.Path:
    path = tmp_cwd / "config.yaml"
    path.write_text(SepTopConfig().model_dump_yaml(sort_keys=False))

    return path


class TestSepTopSetupStage:
    def test_validate_rest_config(self):
        with pytest.raises(
            pydantic.ValidationError, match="REST configuration must be provided if"
        ):
            SepTopSetupStage(apply_rest=True, rest_config=None)


class TestSepTopStates:
    def test_validate_lambda_lengths(self):
        with pytest.raises(
            pydantic.ValidationError, match="fields must have the same length"
        ):
            SepTopStates(
                lambda_vdw_ligand_1=[0.0],
                lambda_charges_ligand_1=[0.0],
                lambda_vdw_ligand_2=[1.0],
                lambda_charges_ligand_2=[1.0],
                lambda_boresch_ligand_1=None,
                lambda_boresch_ligand_2=None,
                bm_b0=[0.0, 0.5, 1.0],
            )


def test_load_config(mock_config_path):
    config = load_config(mock_config_path)
    assert isinstance(config, SepTopConfig)
