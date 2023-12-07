"""Common pydantic validators."""
import ast
import functools
import operator
import pathlib
import re
import typing

import openmm.unit
import pydantic
import pydantic_core
import yaml

_UNIT_LOOKUP = {}

for __unit in openmm.unit.__dict__.values():
    if isinstance(__unit, openmm.unit.Unit) and not __unit == openmm.unit.ampere:
        _UNIT_LOOKUP[__unit.get_symbol()] = __unit
        _UNIT_LOOKUP[__unit.get_name()] = __unit

_UNIT_LOOKUP["amu"] = openmm.unit.atomic_mass_unit
del __unit


IncEx = set[int] | set[str] | dict[int, typing.Any] | dict[str, typing.Any] | None


def openmm_quantity_to_str(value: openmm.unit.Quantity) -> str:
    unit = value.unit
    value = value.value_in_unit(unit)

    # we reverse the order of the bases so that we usually get kcal mol**-1 A**2 rather
    # than eg. A**2 mol**-1 kcal
    bases = list(reversed([*unit.iter_base_or_scaled_units()]))

    components = [
        (
            None if i == 0 else "*",
            base.symbol + ("" if exponent == 1 else f"**{int(exponent)}"),
        )
        for i, (base, exponent) in enumerate(bases)
    ]

    if unit == openmm.unit.dimensionless:
        components = []

    unit_str = " ".join(
        v for component in components for v in component if v is not None
    )
    return f"{value} {unit_str}" if len(unit_str) > 0 else f"{value}"


def _openmm_quantity_from_str(value: str) -> openmm.unit.Quantity:
    def ast_parse(node: ast.expr):
        operators = {
            ast.Pow: operator.pow,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.USub: operator.neg,
        }

        if isinstance(node, ast.Name):
            if node.id not in _UNIT_LOOKUP:
                raise KeyError(f"unit could not be found: {node.id}")
            return _UNIT_LOOKUP[node.id]
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.UnaryOp):
            return operators[type(node.op)](ast_parse(node.operand))
        elif isinstance(node, ast.BinOp):
            return operators[type(node.op)](ast_parse(node.left), ast_parse(node.right))
        else:
            raise NotImplementedError(node)

    value = value.strip()
    value_match = re.match(r"^([0-9.\-+]+)[ ]*[a-zA-Z(\[]", value)

    if value_match:
        split_idx = value_match.regs[-1][-1]
        value = f"{value[:split_idx]} * {value[split_idx:]}"

    return ast_parse(ast.parse(value, mode="eval").body)


def _quantity_validator(
    value: str | openmm.unit.Quantity,
    expected_units: openmm.unit.Unit,
) -> openmm.unit.Quantity:
    if isinstance(value, str):
        value = _openmm_quantity_from_str(value)

    assert isinstance(value, openmm.unit.Quantity), f"invalid type - {type(value)}"

    try:
        return value.in_units_of(expected_units)
    except TypeError as e:
        raise ValueError(
            f"invalid units {value.unit} - expected {expected_units}"
        ) from e


class _OpenMMQuantityAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: typing.Any,
        _handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.core_schema.CoreSchema:
        from_value_schema = pydantic_core.core_schema.no_info_plain_validator_function(
            lambda x: x
        )

        return pydantic_core.core_schema.json_or_python_schema(
            json_schema=from_value_schema,
            python_schema=from_value_schema,
            serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
                openmm_quantity_to_str
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        _core_schema: pydantic_core.core_schema.CoreSchema,
        handler: pydantic.GetJsonSchemaHandler,
    ) -> "pydantic.json_schema.JsonSchemaValue":
        return handler(pydantic_core.core_schema.str_schema())


class _OpenMMQuantityMeta(type):
    def __getitem__(cls, item: openmm.unit.Unit):
        validator = functools.partial(_quantity_validator, expected_units=item)
        return typing.Annotated[
            openmm.unit.Quantity,
            _OpenMMQuantityAnnotation,
            pydantic.BeforeValidator(validator),
        ]


class OpenMMQuantity(openmm.unit.Quantity, metaclass=_OpenMMQuantityMeta):
    """A pydantic safe OpenMM quantity type validates unit compatibility."""


if typing.TYPE_CHECKING:
    OpenMMQuantity = openmm.unit.Quantity  # noqa: F811


class BaseModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        extra="forbid", validate_default=True, validate_assignment=True
    )

    def model_dump_yaml(self, output_path: pathlib.Path | None = None, **kwargs) -> str:
        """Dump the model to a YAML representation.

        Args:
            output_path: The (optional) path to save the YAML representation to.

        Returns:
            The YAML representation.
        """

        model_yaml = yaml.safe_dump(self.model_dump(), **kwargs)

        if output_path is not None:
            output_path.parent.mkdir(exist_ok=True, parents=True)
            output_path.write_text(model_yaml)

        return model_yaml
