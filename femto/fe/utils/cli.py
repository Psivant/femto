"""Utilities for defining CLIs"""

import logging
import pathlib
import types
import typing

import click
import cloup
import pydantic_core

import femto.fe.utils.queue

INPUT_PATH = click.Path(
    exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path
)
OUTPUT_PATH = click.Path(
    exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path
)

INPUT_DIR = click.Path(
    exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path
)
OUTPUT_DIR = click.Path(
    exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path
)


DEFAULT_MAIN_OPTIONS = [
    click.option(
        "-v",
        "--verbose",
        "log_level",
        help="Log debug messages",
        flag_value=logging.DEBUG,
    ),
    click.option(
        "-s",
        "--silent",
        "log_level",
        help="Log only warning messages",
        flag_value=logging.WARNING,
    ),
]
"""The default set of click options to expose on the main command groups."""

DEFAULT_INPUTS = [
    cloup.option(
        "--root-dir",
        type=INPUT_DIR,
        help="The directory containing input files in the standard layout.",
        default=pathlib.Path(""),
    ),
    cloup.option(
        "--edges",
        "edges_path",
        type=INPUT_PATH,
        help="The file defining the alchemical edges to run. By default CLI "
        "will check for a `edges.yaml` or `Morph.in` file in the root "
        "directory",
    ),
    cloup.option("--ligand-1", type=str, help="The name of the first ligand"),
    cloup.option("--ligand-2", type=str, help="The name of the second ligand"),
]
DEFAULT_INPUTS_GROUP_NAME = "Input directory"
DEFAULT_INPUTS_GROUP_HELP = None
DEFAULT_INPUTS_GROUP = cloup.option_group(
    DEFAULT_INPUTS_GROUP_NAME, *DEFAULT_INPUTS, help=DEFAULT_INPUTS_GROUP_HELP
)
DEFAULT_OUTPUTS = [
    cloup.option(
        "--output-dir",
        type=OUTPUT_DIR,
        required=True,
        help="The root directory to store any outputs in.",
    ),
    cloup.option(
        "--report-dir",
        type=OUTPUT_DIR,
        help="The (optional) directory to report statistics to.",
    ),
]
DEFAULT_OUTPUTS_GROUP_NAME = "Outputs"
DEFAULT_OUTPUTS_GROUP_HELP = None
DEFAULT_OUTPUTS_GROUP = cloup.option_group(
    DEFAULT_OUTPUTS_GROUP_NAME, *DEFAULT_OUTPUTS, help=DEFAULT_OUTPUTS_GROUP_HELP
)

DEFAULT_RECEPTOR_PATHS = [
    cloup.option(
        "--receptor-coords",
        type=INPUT_PATH,
        help="The path to the receptor coordinates (.rst7).",
    ),
    cloup.option(
        "--receptor-params",
        type=INPUT_PATH,
        help="The (optional) path to the receptor parameters (.parm7).",
    ),
]
DEFAULT_RECEPTOR_PATHS_GROUP_NAME = "Receptor paths"
DEFAULT_RECEPTOR_PATHS_GROUP_HELP = None
DEFAULT_RECEPTOR_PATHS_GROUP = cloup.option_group(
    DEFAULT_RECEPTOR_PATHS_GROUP_NAME,
    *DEFAULT_RECEPTOR_PATHS,
    help=DEFAULT_RECEPTOR_PATHS_GROUP_HELP,
)
DEFAULT_RECEPTOR_OPTIONS = []
DEFAULT_RECEPTOR_OPTIONS_GROUP_NAME = "Receptor options"
DEFAULT_RECEPTOR_OPTIONS_GROUP_HELP = None

DEFAULT_LIGAND_PATHS = [
    cloup.option(
        "--ligand-1-coords",
        type=INPUT_PATH,
        help="The path to the coordinates (.rst7) of the first ligand.",
    ),
    cloup.option(
        "--ligand-1-params",
        type=INPUT_PATH,
        help="The path to the parameters (.parm7) of the first ligand.",
    ),
    cloup.option(
        "--ligand-2-coords",
        type=INPUT_PATH,
        default=None,
        help="The path to the coordinates (.rst7) of the second ligand.",
    ),
    cloup.option(
        "--ligand-2-params",
        type=INPUT_PATH,
        help="The path to the parameters (.parm7) of the second ligand.",
    ),
]
DEFAULT_LIGAND_PATHS_GROUP_NAME = "Ligand paths"
DEFAULT_LIGAND_PATHS_GROUP_HELP = None
DEFAULT_LIGAND_PATHS_GROUP = cloup.option_group(
    DEFAULT_LIGAND_PATHS_GROUP_NAME,
    *DEFAULT_LIGAND_PATHS,
    help=DEFAULT_LIGAND_PATHS_GROUP_HELP,
)
DEFAULT_LIGAND_OPTIONS = [
    cloup.option(
        "--ligand-1-ref-atoms",
        type=(str, str, str),
        required=False,
        help="Three (optional) AMBER style queries that select the atoms of the first "
        "ligand to align using.",
    ),
    cloup.option(
        "--ligand-2-ref-atoms",
        type=(str, str, str),
        required=False,
        help="Three (optional) AMBER style queries that select the atoms of the second "
        "ligand to align using.",
    ),
]
DEFAULT_LIGAND_OPTIONS_GROUP_NAME = "Ligand options"
DEFAULT_LIGAND_OPTIONS_GROUP_HELP = None
DEFAULT_LIGAND_OPTIONS_GROUP = cloup.option_group(
    DEFAULT_LIGAND_OPTIONS_GROUP_NAME,
    *DEFAULT_LIGAND_OPTIONS,
    help=DEFAULT_LIGAND_OPTIONS_GROUP_HELP,
)

DEFAULT_SUBMIT_OPTIONS = [
    cloup.option(
        "--wait",
        is_flag=True,
        help="Wait for the submitted jobs to complete before exiting. If set, any "
        "all submitted jobs will be cancelled if the command is terminated.",
        type=bool,
    )
]
DEFAULT_SUBMIT_OPTIONS_GROUP_NAME = "Submit options"
DEFAULT_SUBMIT_OPTIONS_GROUP_HELP = None


def generate_slurm_cli_options(job_name: str | None, required: bool = False) -> list:
    """A helper function to generate the default set of options to add to a CLI
    function that will submit SLURM jobs.

    Args:
        job_name: The default job name to use.
        required: Whether options without defaults should be required or not.

    Returns
        A list of click options.
    """
    options = []

    for (
        field_name,
        field,
    ) in femto.fe.utils.queue.SLURMOptions.model_fields.items():
        description = field.description

        default_value = field.default
        has_default = (
            default_value is not Ellipsis
            and default_value != pydantic_core.PydanticUndefined
        )

        if job_name is not None and field_name == "job_name":
            has_default = True
            default_value = job_name

        flag = f"--slurm-{field_name}".replace("_", "-").replace("-n-", "-")
        field_type = field.annotation

        if isinstance(field_type, types.UnionType):
            type_args = typing.get_args(field_type)
            assert len(type_args) == 2 and types.NoneType in type_args

            required = False
            field_type = (
                type_args[0]  # noqa: E721
                if type_args[1] is types.NoneType  # noqa: E721
                else type_args[1]  # noqa: E721
            )

        options.append(
            click.option(
                flag,
                f"slurm_{field_name}",
                help=description,
                type=field_type,
                required=required and not has_default,
                default=None if not has_default else default_value,
                show_default=has_default,
            )
        )

    return options


def add_options(options):
    """Apply a list of options / arguments to a function"""

    options = options if isinstance(options, typing.Iterable) else [options]

    def _apply_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _apply_options


def validate_mutually_exclusive_groups(
    context: cloup.Context,
    group_1_titles: list[str] | str,
    group_2_titles: list[str] | str,
    optional_fields: set[str] | None = None,
) -> tuple[bool, bool]:
    """Checks whether the user tried to specify options from two mutually exclusive
    sets of option groups.

    Args:
        context: The click context.
        group_1_titles: The titles of the first groups.
        group_2_titles: The titles of the second groups.
        optional_fields: A set of fields that are optional and do not have default
            values.

    Returns:
        Whether the user specified options from group 1 and group 2, respectively.
    """
    optional_fields = set() if optional_fields is None else optional_fields

    group_1_titles = (
        [group_1_titles] if isinstance(group_1_titles, str) else group_1_titles
    )
    group_2_titles = (
        [group_2_titles] if isinstance(group_2_titles, str) else group_2_titles
    )

    command = typing.cast(cloup.Command, context.command)

    group_1_matches = [
        group for group in command.option_groups if group.title in group_1_titles
    ]
    assert len(group_1_matches) == len(
        group_1_titles
    ), f"found {len(group_1_matches)} group 1 matches."

    group_2_matches = [
        group for group in command.option_groups if group.title in group_2_titles
    ]
    assert len(group_2_matches) == len(
        group_2_titles
    ), f"found {len(group_2_matches)} group 2 matches."

    group_1 = group_1_matches[0]
    group_1_options = [option.name for option in group_1.options]
    group_2 = group_2_matches[0]
    group_2_options = [option.name for option in group_2.options]

    found_group_1_options = any(
        context.get_parameter_source(option) != click.core.ParameterSource.DEFAULT
        for option in group_1_options
    )
    found_group_2_options = any(
        context.get_parameter_source(option) != click.core.ParameterSource.DEFAULT
        for option in group_2_options
    )

    if found_group_1_options and found_group_2_options:
        raise click.UsageError(
            f"Options from the {group_1_titles} and {group_2_titles} option groups are "
            f"mutually exclusive."
        )
    if not found_group_1_options and not found_group_2_options:
        raise click.UsageError(
            f"Options from either the {group_1_titles} or {group_2_titles} option "
            f"groups must be specified."
        )

    required_options = group_1.options if found_group_1_options else group_2.options

    for option in required_options:
        value = context.params[option.name]

        if option.name in optional_fields or value is not None:
            continue

        raise click.MissingParameter(ctx=context, param=option)

    return found_group_1_options, found_group_2_options


def configure_logging(log_level: int | None):
    """Set up basic logging for the CLI, silencing any overly verbose modules (e.g.
    parmed).

    Args:
        log_level: The log level to use.
    """

    logging.basicConfig(
        level=log_level if log_level is not None else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%c",
    )
    logging.getLogger("parmed").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("pymbar").setLevel(logging.WARNING)
    logging.getLogger("jax").setLevel(logging.ERROR)
