"""Command-line interface for ``femto``."""

import cloup

import femto.fe.atm._cli
import femto.fe.septop._cli
import femto.fe.utils.cli
import femto.md.utils.logging
import femto.md.utils.mpi

_LOGGER = femto.md.utils.logging.get_parent_logger(__package__)


@cloup.group()
@femto.fe.utils.cli.add_options(femto.fe.utils.cli.DEFAULT_MAIN_OPTIONS)
def main(log_level: int):
    femto.fe.utils.cli.configure_logging(log_level)

    if femto.md.utils.mpi.is_rank_zero():
        _LOGGER.info(f"Running femto version={femto.fe.__version__}")


main.add_command(femto.fe.atm._cli.main)
main.add_command(femto.fe.septop._cli.main)
