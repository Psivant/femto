import click
import click.testing
import cloup
import pytest

import femto.fe.utils.cli


def test_generate_slurm_cli_options():
    options = femto.fe.utils.cli.generate_slurm_cli_options(None, True)

    @click.command
    @femto.fe.utils.cli.add_options(options)
    def mock_func(**kwargs):
        pass

    runner = click.testing.CliRunner()
    result = runner.invoke(mock_func, ["--help"])

    assert result.exit_code == 0
    assert (
        result.output
        == """Usage: mock-func [OPTIONS]

Options:
  --slurm-nodes INTEGER          The number of nodes to request.  [required]
  --slurm-tasks INTEGER          The number of tasks to request.  [required]
  --slurm-cpus-per-task INTEGER  The cpus per task.  [default: 1]
  --slurm-gpus-per-task INTEGER  The number of gpus to request per task.
                                 [default: 1]
  --slurm-walltime TEXT          The maximum walltime.  [default: 5-0]
  --slurm-partition TEXT         The partition to run on.  [required]
  --slurm-account TEXT           The account to use.
  --slurm-job-name TEXT          The job name to use.  [default: femto]
  --slurm-reservation TEXT       The reservation to use.
  --help                         Show this message and exit.
"""
    )


def test_add_options():
    def mock_func():
        pass

    command = click.command(mock_func)
    command = femto.fe.utils.cli.add_options(
        [click.option("--a"), click.option("--b")]
    )(command)

    assert len(command.params) == 2
    assert command.params[0].name == "b"
    assert command.params[1].name == "a"


@pytest.mark.parametrize(
    "args, expected_return, expected_exit_code, expected_contains",
    [
        (["--a", "1", "--b", "2"], (True, False), 0, ""),
        (["--c", "3"], (False, True), 0, ""),
        ([], None, 2, "Options from either the"),
        (["--a", "1", "--c", "3"], None, 2, "mutually exclusive"),
        (["--a", "1"], None, 2, "Missing option '--b'"),
    ],
)
def test_validate_mutually_exclusive_groups(
    args, expected_return, expected_exit_code, expected_contains, mocker
):
    spied_fn = mocker.spy(femto.fe.utils.cli, "validate_mutually_exclusive_groups")

    @cloup.command()
    @cloup.option_group("Group 1", cloup.option("--a"), cloup.option("--b"))
    @cloup.option_group(
        "Group 2", cloup.option("--c"), cloup.option("--d", default="v")
    )
    @cloup.pass_context
    def mock_func(ctx, a, b, c, d):
        femto.fe.utils.cli.validate_mutually_exclusive_groups(ctx, "Group 1", "Group 2")

    runner = click.testing.CliRunner()
    result = runner.invoke(mock_func, args)

    assert result.exit_code == expected_exit_code
    assert expected_contains in result.output

    if expected_return is not None:
        assert spied_fn.spy_return == expected_return
