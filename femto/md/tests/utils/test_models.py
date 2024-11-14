import openmm.unit
from pydantic_units import OpenMMQuantity

from femto.md.utils.models import BaseModel


def test_model_dump_yaml(tmp_cwd):
    class MockModel(BaseModel):
        a: OpenMMQuantity[openmm.unit.angstrom] = 1.0 * openmm.unit.nanometers
        b: OpenMMQuantity[openmm.unit.kelvin] = 2.0 * openmm.unit.kelvin

    expected_file = tmp_cwd / "model.yaml"
    actual = MockModel().model_dump_yaml(expected_file)

    expected = "a: 10.0 A\nb: 2.0 K\n"

    assert actual == expected
    assert expected_file.read_text() == actual
