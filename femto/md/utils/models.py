"""Common pydantic helpers."""

import pathlib

import pydantic
import yaml


class BaseModel(pydantic.BaseModel):
    """A base class for ``pydantic`` based models."""

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
