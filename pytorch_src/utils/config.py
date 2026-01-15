import os
import pathlib
import argparse
import logging
from datetime import datetime
from functools import reduce
from operator import getitem
from utils.utils import load_json, write_json, import_attr, get_logger
from typing import Optional, Any


class ConfigParser:
    """
    Class to parse configuration json file. Handles hyperparameters for training and initializations of modules.
    """

    def __init__(
        self,
        config: dict,
        resume: Optional[pathlib.Path | str] = None,
        modification: Optional[dict] = None,
        run_id: Optional[str] = None,
    ):
        """
        Initialize the ConfigParser class.

        Parameters
        ----------
        config : dict
            Dict containing configurations, hyperparameters for training. contents of `config.json` file for example
        resume : pathlib.Path or str, optional
            String, path to the checkpoint being loaded.
        modification : dict, optional
            Dict keychain:value, specifying position values to be replaced from config dict.
        run_id : str, optional
            Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default.

        """
        self._config = _update_config(config, modification)
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        save_dir = pathlib.Path(self.config['save_dir'])

        exper_name = self.config['name']
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'logs' / exper_name / run_id

        # update the trainer config
        self._config["trainer"]["save_dir"] = str(self.save_dir)
        self._config["trainer"]["log_dir"] = str(self.log_dir)

        # make directory for saving checkpoints and log.
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')

        self.log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    def __getitem__(self, key: str) -> object:
        """Access items like a dict."""
        return self._config.get(key, None)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._config

    def __iter__(self):
        """Iterate over config keys."""
        return iter(self._config)

    def __getattr__(self, name: str) -> Any:
        """Return attribute-style access to config keys."""
        if name.startswith('_'):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        if name in self._config:
            return self._config[name]

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def keys(self) -> list[str]:
        """Return config keys."""
        return self._config.keys()

    def values(self) -> list[object]:
        """Return config values."""
        return self._config.values()

    def items(self) -> list[tuple[str, object]]:
        """Return config items."""
        return self._config.items()

    def get(self, key: str, default=None) -> object:
        """Get items like a dict with default value."""
        return self._config.get(key, default)

    @classmethod
    def from_args(
        cls, args: argparse.ArgumentParser, options: Optional[list[object]] = None
    ) -> "ConfigParser":
        """
        Initialize this class from some cli arguments. Used in train, test.

        Parameters
        ----------
        args : argparse.ArgumentParser
            Command line arguments including:
            - config : str
                Path to config file.
            - resume : str
                Path to the checkpoint being loaded.
            - device : str
                CUDA_VISIBLE_DEVICES setting.

        options : Optional[list of CustomArgs]
            List of CustomArgs (defined in utils) used to specify additional command line options, see Examples.

        Returns
        -------
        ConfigParser
            An instance of ConfigParser initialized from the provided arguments.

        Examples
        --------
        ```python
        parser = argparse.ArgumentParser(description='Training Config')
        parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
        parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
        parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
        CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
        options = [
            CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
            CustomArgs(['--bs', '--batch_size'], type=int, target='dataloader;args;batch_size')
        ]
        cfg = ConfigParser.from_args(parser.parse_args(), options)

        """
        for opt in options or []:
            if not all(hasattr(opt, attr) for attr in ['flags', 'type', 'target']):
                raise AttributeError(
                    "CustomArgs must have 'flags', 'type', and 'target' attributes."
                )
            if not isinstance(opt.flags, list):
                raise TypeError(
                    "'flags' attribute of CustomArgs must be a list of strings."
                )
            args.add_argument(*opt.flags, default=None, type=opt.type)

        args = args.parse_args()

        device = getattr(args, 'device', None)
        resume = getattr(args, 'resume', None)
        if device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = device

        if resume is not None:
            resume = pathlib.Path(resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = pathlib.Path(args.config)

        config = load_json(cfg_fname)
        if args.config and resume:
            config.update(load_json(args.config))

        # parse custom cli options into dictionary
        modification = {
            opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options
        }
        return cls(config, resume, modification)

    def get_logger(self, name: str, verbosity=2, **kwargs) -> logging.Logger:
        """
        Get a logger instance.

        Parameters
        ----------
        name : str
            Name of the logger.
        verbosity : int
            Verbosity level, 0: WARNING, 1: INFO, 2: DEBUG

        Returns
        -------
        logging.Logger
            Configured logger instance.
        """
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(
            verbosity, self.log_levels.keys()
        )
        assert verbosity in self.log_levels, msg_verbosity
        logger = get_logger(name, level=self.log_levels[verbosity], **kwargs)
        return logger

    def init_obj(self, field, *args, **kwargs) -> object:
        """
        Initialize an object from the config file with the following structure.

        "field_name": {
            "module": "module.name",
            "type": "ClassName",
            "args": {
                "arg1": value1,
                "arg2": value2,
                ...
            }
        }

        Parameters
        ----------
        field : str
            The field name in the config file.
        *args :
            Positional arguments to pass to the class constructor.
        **kwargs :
            Additional keyword arguments not specified in the config file. These will be merged with those in the config file, but cannot overwrite them.

        Returns
        -------
        object
            An instance of the specified class initialized with the provided arguments.

        Raises
        ------
        KeyError
            If the specified field is not found in the config file.
            If there are conflicting keys between the config file and the provided kwargs.
        """
        if self[field] is None:
            raise KeyError(f"Field '{field}' not found in config.")

        module = self[field].get("module", None)
        class_name = self[field].get("type", None)

        if module is None and class_name is None:
            raise KeyError(f"module and type not specified for '{field}' in config.")

        cfg_kwargs = dict(self[field].get("args", {}))

        overlap = [k for k in kwargs if k in cfg_kwargs]
        if overlap:
            raise KeyError(
                f"Overwriting kwargs given in config file is not allowed. "
                f"Conflicting keys: {overlap}"
            )

        cfg_kwargs.update(kwargs)

        obj, _ = import_attr(class_name, module, *args, **cfg_kwargs)
        return obj

    # setting read-only attributes
    @property
    def config(self) -> dict:
        """Returns the config dictionary."""
        return self._config

    @property
    def save_dir(self) -> pathlib.Path:
        """Returns the save directory path."""
        return self._save_dir

    @property
    def log_dir(self) -> pathlib.Path:
        """Returns the log directory path."""
        return self._log_dir


# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    """
    Update configuration dict by replacing values from modification dict.
    """
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags) -> str:
    """
    Get the option name from command line flags.
    """
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys) -> object:
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
