"""Utilities to setup logging"""

import logging


def get_public_parent_module(name: str) -> str:
    """Get the first public parent module of a module.

    Args:
        name: The full name (e.g. ``'mod_a.mod_b._mod_c'``) of the module.

    Returns:
        The full name of the first public parent module (e.g. ``'mod_a.mod_b'``).
    """

    parts = name.split(".")

    while parts and parts[-1].startswith("_"):
        parts.pop()

    return ".".join(parts)


def get_parent_logger(name: str) -> logging.Logger:
    """Returns the logger of the first public parent module of a module.

    Args:
        name: The full name (e.g. ``'mod_a.mod_b._mod_c'``) of the module.

    Returns:
        The logger.
    """

    return logging.getLogger(get_public_parent_module(name))
