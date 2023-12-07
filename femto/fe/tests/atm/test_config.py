import pathlib

import openmm.unit
import pydantic
import pytest

from femto.fe.atm._config import ATMConfig, ATMSetupStage, ATMStates, load_config


@pytest.fixture
def mock_config_path(tmp_cwd) -> pathlib.Path:
    path = tmp_cwd / "config.yaml"
    path.write_text(ATMConfig().model_dump_yaml(sort_keys=False))

    return path


class TestATMSetupStage:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (38.0 * openmm.unit.angstrom, "displacement: 38.0 A"),
            (
                [38.0 * openmm.unit.angstrom] * 3,
                "displacement:\n- 38.0 A\n- 38.0 A\n- 38.0 A",
            ),
        ],
    )
    def test_serialize_displacement(self, value, expected):
        config = ATMSetupStage(displacement=value)
        config_yaml = config.model_dump_yaml(sort_keys=False)

        assert expected in config_yaml

    def test_validate_rest_config(self):
        with pytest.raises(
            pydantic.ValidationError, match="REST configuration must be provided if"
        ):
            ATMSetupStage(apply_rest=True, rest_config=None)


class TestATMStates:
    def test_validate_lambda_lengths(self):
        with pytest.raises(
            pydantic.ValidationError, match="fields must have the same length"
        ):
            ATMStates(lambda_1=[0.0])


def test_load_config(mock_config_path):
    config = load_config(mock_config_path)
    assert isinstance(config, ATMConfig)
