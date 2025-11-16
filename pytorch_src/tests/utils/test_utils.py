import pytest
from utils.utils import *


def test_prepare_device_invalid_negative_gpu():
    """Test that ValueError is raised for negative n_gpu_use"""
    with pytest.raises(ValueError) as exc_info:
        prepare_device(-1)
    assert "n_gpu_use must be non-negative" in str(exc_info.value)


def test_prepare_device_no_gpu_requested():
    """Test when n_gpu_use is 0, should return CPU device"""
    device, list_ids = prepare_device(0)
    assert device.type == "cpu"
    assert list_ids == []


def test_prepare_device_gpu_available(monkeypatch):
    """Test when GPU is available and requested"""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    device, list_ids = prepare_device(2)
    assert device.type == "cuda"
    assert list_ids == [0, 1]


def test_import_attr_successful_import():
    """Test successful import of an attribute from a single module"""
    attr, mod_name = import_attr("Path", "pathlib")
    assert mod_name == "pathlib"
    assert attr.__module__ == "pathlib"


def test_import_attr_multiple_modules():
    """Test import from multiple modules, should return first match"""
    attr, mod_name = import_attr("Path", ["os", "pathlib"])
    assert mod_name == "pathlib"


def test_import_attr_second_module():
    """Test import when attribute exists in second module only"""
    attr, mod_name = import_attr("OrderedDict", ["os", "collections"])
    assert mod_name == "collections"


def test_import_attr_nonexistent_attribute():
    """Test that ImportError is raised when attribute doesn't exist"""
    with pytest.raises(ImportError) as exc_info:
        import_attr("NonExistentClass", ["os", "pathlib"])
    assert "Could not find attribute 'NonExistentClass'" in str(exc_info.value)
    assert "Attribute not found" in str(exc_info.value)


def test_import_attr_nonexistent_module():
    """Test that ImportError is raised when module doesn't exist"""
    with pytest.raises(ImportError) as exc_info:
        import_attr("Path", ["nonexistent_module"])
    assert "Could not find attribute 'Path'" in str(exc_info.value)


def test_import_attr_empty_modules_list():
    """Test that ValueError is raised when no modules provided"""
    with pytest.raises(ValueError) as exc_info:
        import_attr("Path", [])
    assert "No modules provided" in str(exc_info.value)


def test_import_attr_mixed_valid_invalid_modules():
    """Test import with mix of valid and invalid modules"""
    attr, mod_name = import_attr("Path", ["invalid_module", "pathlib", "os"])
    assert mod_name == "pathlib"


def test_import_attr_string_single_module():
    """Test that single module as string works correctly"""
    attr, mod_name = import_attr("OrderedDict", "collections")
    assert mod_name == "collections"

def test_prepare_device_no_gpu_requested():
    """Test when n_gpu_use is 0, should return CPU device"""
    device, list_ids = prepare_device(0)
    assert device.type == "cpu"
    assert list_ids == []

def test_prepare_device_gpu_available(monkeypatch):
    """Test when GPU is available and requested"""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    device, list_ids = prepare_device(2)
    assert device.type == "cuda"
    assert list_ids == [0, 1]

def test_prepare_device_request_more_than_available(monkeypatch, capsys):
    """Test when requesting more GPUs than available"""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    device, list_ids = prepare_device(4)
    assert device.type == "cuda"
    assert list_ids == [0, 1]
    captured = capsys.readouterr()
    assert "Warning: The number of GPU's configured to use is 4" in captured.out

def test_prepare_device_no_gpu_available_but_requested(monkeypatch, capsys):
    """Test when GPU is requested but none available"""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    device, list_ids = prepare_device(2)
    assert device.type == "cpu"
    assert list_ids == []
    captured = capsys.readouterr()
    assert "Warning: There's no GPU available on this machine" in captured.out

def test_prepare_device_single_gpu(monkeypatch):
    """Test when requesting single GPU"""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    device, list_ids = prepare_device(1)
    assert device.type == "cuda"
    assert list_ids == [0]

def test_prepare_device_exact_gpu_match(monkeypatch):
    """Test when requested GPUs exactly matches available"""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    device, list_ids = prepare_device(3)
    assert device.type == "cuda"
    assert list_ids == [0, 1, 2]
