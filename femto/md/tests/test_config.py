import pydantic

import femto.md.config
import femto.md.utils.models


def test_merge_configs():
    class ConfigB(femto.md.utils.models.BaseModel):
        a: int = pydantic.Field(3)
        b: list[int] = pydantic.Field([1, 2])

        g: str = pydantic.Field("hi")

    config_a = {"a": 1, "b": None, "c": {"d": 2, "e": None}}
    config_b = ConfigB()
    config_c = {"a": 4, "b": [3], "c": {"e": {"f": 5}}}

    config = femto.md.config.merge_configs(config_a, config_b, config_c)

    expected_config = {"a": 4, "b": [3], "c": {"d": 2, "e": {"f": 5}}, "g": "hi"}
    assert config == expected_config
