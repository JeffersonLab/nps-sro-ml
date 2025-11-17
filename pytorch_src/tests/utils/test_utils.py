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


def test_import_attr_with_args():
    """Test import_attr with args and no kwargs for callable attributes"""
    attr, mod_name = import_attr("defaultdict", "collections", int)
    assert mod_name == "collections"
    assert isinstance(attr, dict)
    assert attr['missing_key'] == 0  # default factory is int, so default is 0


def test_import_attr_with_kwargs():
    """Test import_attr with no args and kwargs for callable attributes"""
    attr, mod_name = import_attr(
        "namedtuple", "collections", typename='Point', field_names=['x', 'y']
    )
    assert mod_name == "collections"
    Point = attr
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2


def test_import_attr_with_args_and_kwargs():
    """Test import_attr with both args and kwargs for callable attributes"""
    attr, mod_name = import_attr("Counter", "collections", 'abracadabra', **{'a': 5})
    assert mod_name == "collections"
    counter = attr
    assert counter['a'] == 10  # 'a' appears 5 times in 'abracadabra' + 5 from kwargs
    assert counter['b'] == 2
    assert counter['r'] == 2
    assert counter['c'] == 1
    assert counter['d'] == 1


def test_import_attr_no_args_kwargs_on_callable():
    """Test import_attr without args and kwargs for callable attributes"""
    attr, mod_name = import_attr("copy", "copy")
    assert mod_name == "copy"
    copy_func = attr
    original = [1, 2, 3]
    copied = copy_func(original)
    assert copied == original
    assert copied is not original  # Ensure it's a shallow copy


def test_import_attr_no_args_kwargs_on_non_callable():
    """Test import_attr without args and kwargs for non-callable attributes"""
    attr, mod_name = import_attr("pi", "math")
    assert mod_name == "math"
    assert pytest.approx(attr) == 3.141592653589793


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
