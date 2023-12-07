"""A comprehensive toolkit for predicting free energies"""

import logging

from . import _version


class __FilterSimTKDeprecation(logging.Filter):
    """Disable the deprecation warning from SimTK triggered by ParmEd which spams any
    CLI output / logs."""

    def filter(self, record):
        return "importing 'simtk.openmm' is deprecated" not in record.getMessage()


__root_logger = logging.getLogger()
__root_logger.addFilter(__FilterSimTKDeprecation())

del __root_logger
del __FilterSimTKDeprecation

__version__ = _version.get_versions()["version"]
