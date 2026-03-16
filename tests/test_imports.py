"""Test that all main modules can be imported."""

import pytest


def test_import_balancepy():
    """Test that main package imports successfully."""
    import balancepy
    assert balancepy.__version__ == "0.1.1"


def test_import_modules():
    """Test that all submodules import successfully."""
    import balancepy
    
    # Check that all documented modules are accessible
    assert hasattr(balancepy, "anaropia")
    assert hasattr(balancepy, "biomechanics")
    assert hasattr(balancepy, "timeseries")
    assert hasattr(balancepy, "frequency")
    assert hasattr(balancepy, "make_stimulus")
    assert hasattr(balancepy, "data_class")
    assert hasattr(balancepy, "base_model")
    assert hasattr(balancepy, "parameter")


def test_all_export():
    """Test that __all__ is properly defined."""
    import balancepy
    assert hasattr(balancepy, "__all__")
    assert len(balancepy.__all__) > 0
