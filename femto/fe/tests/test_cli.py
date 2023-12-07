import typing

import click.testing
import pytest

import femto.fe._cli


@pytest.fixture
def click_runner() -> click.testing.CliRunner:
    runner = click.testing.CliRunner()
    yield runner


def test_merge_config(click_runner, tmp_cwd):
    result: click.testing.Result = click_runner.invoke(
        typing.cast(click.Command, femto.fe._cli.main)
    )
    assert result.exit_code == 0
