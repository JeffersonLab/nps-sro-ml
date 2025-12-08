import torch
import json
import pathlib
import logging
import importlib
from collections import OrderedDict
from typing import Optional, List, Any, Tuple
from collections import OrderedDict


def prepare_device(n_gpu_use: int) -> Tuple[torch.device, List[int]]:
    """
    Prepare for GPU device if available and get gpu device indices which are used for DataParallel.

    Parameters
    ----------
    n_gpu_use : int
        Number of GPUs to use.
    """
    if n_gpu_use < 0:
        raise ValueError("n_gpu_use must be non-negative")

    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def load_json(fname: str | pathlib.Path) -> OrderedDict:
    """
    Load JSON file and return as an OrderedDict.

    Parameters
    ----------
    fname : str | pathlib.Path
        Path to the JSON file.

    Returns
    -------
    OrderedDict
        Contents of the JSON file as an OrderedDict.
    """
    fname = pathlib.Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: dict, fname: str | pathlib.Path):
    """
    Write a dictionary to a JSON file.

    Parameters
    ----------
    content : dict
        Dictionary to write to the JSON file.
    fname : str | pathlib.Path
        Path to the JSON file.
    """
    fname = pathlib.Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def import_attr(
    attr_name: str, avail_modules: List[str] | str, *args, **kwargs
) -> Tuple[Any, str]:
    """
    Dynamically import an attribute (class, function, variable) from a list of available modules. By default, returns the first successfully imported attribute. If the attribute is not found in any of the provided modules, raises an ImportError with detailed information about the attempts.

    Parameters
    ----------
    attr_name : str
        Name of the attribute to import.
    avail_modules : List[str] | str
        List of module names (as strings) to search for the attribute.

    Returns
    -------
    Tuple[Any, str]
        The imported attribute and module name.

    Raises
    ------
    ImportError
        If the attribute is not found in any of the provided modules.
    Example
    -------
    attr, mod_name = import_attr("Path", ["os", "pathlib"])
    assert attr.__name__ == "Path"
    assert mod_name == "pathlib"
    """
    if isinstance(avail_modules, str):
        avail_modules = [avail_modules]

    if not avail_modules:
        raise ValueError("No modules provided in `avail_modules`.")

    log_msg = {mod: [] for mod in avail_modules}

    for mod_name in avail_modules:
        try:
            module = importlib.import_module(mod_name)
        except ImportError as e:
            log_msg[mod_name].append(str(e))
            continue

        attr = getattr(module, attr_name, None)
        if attr is not None:
            if args or kwargs:
                attr = attr(*args, **kwargs)
            return attr, mod_name
        else:
            log_msg[mod_name].append("Attribute not found.")

    available_str = ", ".join(avail_modules)
    log_details = "\n".join(
        [f"In module '{mod}': {', '.join(errors)}" for mod, errors in log_msg.items()]
    )
    raise ImportError(
        f"Could not find attribute '{attr_name}' in any of: [{available_str}].\n"
        f"Details of import attempts:\n{log_details}"
    )


def get_logger(
    name: str,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s - %(name)s - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    file: Optional[str | pathlib.Path] = None,
) -> logging.Logger:
    """
    Create and configure a logger.

    Parameters
    ----------
    name : str
        Name of the logger.
    level : int, optional
        Logging level (default is logging.INFO).
    fmt : str, optional
        Format of the log messages (default is "%(asctime)s - %(name)s - %(message)s").
    datefmt : str, optional
        Format of the date in log messages (default is "%Y-%m-%d %H:%M:%S").
    file : str | None, optional
        If provided, log messages will be written to this file. If None, logs will be printed to console.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        filename=file,
        filemode="a" if file else None,
    )
    logger = logging.getLogger(name)
    return logger
