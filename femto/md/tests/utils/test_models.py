import random

import numpy
import openmm.unit
import pydantic
import pytest

import femto.md.utils.openmm
from femto.md.utils.models import (
    BaseModel,
    OpenMMQuantity,
    _openmm_quantity_from_str,
    openmm_quantity_to_str,
)


@pytest.mark.parametrize(
    "unit, expected_str",
    [
        (openmm.unit.angstrom, " A"),
        (openmm.unit.angstrom**2, " A**2"),
        (openmm.unit.atomic_mass_unit, " Da"),
        (openmm.unit.kilojoules_per_mole, " kJ * mol**-1"),
        (openmm.unit.nanometers, " nm"),
        (
            openmm.unit.kilojoules_per_mole / openmm.unit.kelvin**2,
            " kJ * mol**-1 * K**-2",
        ),
        (openmm.unit.dimensionless, ""),
    ],
)
def testopenmm_quantity_to_str(unit, expected_str):
    value = random.random()

    expected = f"{value}{expected_str}"
    actual = openmm_quantity_to_str(value * unit)

    assert actual == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1.0", 1.0),
        ("2.0 A", 2.0 * openmm.unit.angstrom),
        ("-2.0 A", -2.0 * openmm.unit.angstrom),
        ("3.0A**2", 3.0 * openmm.unit.angstrom**2),
        (
            "4.0(kcal/(mol*A**2))",
            4.0 * openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom**2,
        ),
        ("5.0  amu", 5.0 * openmm.unit.atomic_mass_unit),
    ],
)
def test_openmm_quantity_from_str(value, expected):
    actual = _openmm_quantity_from_str(value)

    if isinstance(expected, openmm.unit.Unit):
        assert actual == expected
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected)
    else:
        assert femto.md.utils.openmm.is_close(actual, expected)


@pytest.mark.parametrize(
    "input_value",
    [1.0 * openmm.unit.angstrom, 0.1 * openmm.unit.nanometers, "1.0 angstrom"],
)
def test_openmm_unit_type(input_value):
    class MockModel(pydantic.BaseModel):
        value: OpenMMQuantity[openmm.unit.angstrom]

    x = MockModel(value=input_value)

    assert isinstance(x.value, openmm.unit.Quantity)
    assert x.value.unit == openmm.unit.angstrom
    assert numpy.isclose(x.value.value_in_unit(openmm.unit.angstrom), 1.0)

    model_json = x.model_dump_json()
    assert model_json == '{"value":"1.0 A"}'

    model_schema = x.model_json_schema()
    assert model_schema == {
        "properties": {"value": {"title": "Value", "type": "string"}},
        "required": ["value"],
        "title": "MockModel",
        "type": "object",
    }

    y = MockModel.model_validate_json(model_json)
    assert isinstance(y.value, openmm.unit.Quantity)
    assert y.value.unit == openmm.unit.angstrom
    assert numpy.isclose(y.value.value_in_unit(openmm.unit.angstrom), 1.0)


def test_openmm_unit_type_incompatible():
    class MockModel(pydantic.BaseModel):
        value: OpenMMQuantity[openmm.unit.angstrom]

    with pytest.raises(
        pydantic.ValidationError,
        match="invalid units kilocalorie/mole - expected angstrom",
    ):
        MockModel(value=1.0 * openmm.unit.kilocalories_per_mole)


def test_model_dump_yaml(tmp_cwd):
    class MockModel(BaseModel):
        a: OpenMMQuantity[openmm.unit.angstrom] = 1.0 * openmm.unit.nanometers
        b: OpenMMQuantity[openmm.unit.kelvin] = 2.0 * openmm.unit.kelvin

    expected_file = tmp_cwd / "model.yaml"
    actual = MockModel().model_dump_yaml(expected_file)

    expected = "a: 10.0 A\nb: 2.0 K\n"

    assert actual == expected
    assert expected_file.read_text() == actual
