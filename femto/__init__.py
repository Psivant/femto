"""A comprehensive toolkit for predicting free energies"""

import importlib.metadata
import logging


class __FilterSimTKDeprecation(logging.Filter):
    """Disable the deprecation warning from SimTK triggered by ParmEd which spams any
    CLI output / logs."""

    def filter(self, record):
        return "importing 'simtk.openmm' is deprecated" not in record.getMessage()


__root_logger = logging.getLogger()
__root_logger.addFilter(__FilterSimTKDeprecation())

del __root_logger
del __FilterSimTKDeprecation

try:
    __version__ = importlib.metadata.version("femto")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
