import importlib


def test_fe_imported():
    assert importlib.import_module("femto.fe") is not None
