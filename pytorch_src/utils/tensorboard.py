import pathlib
import logging
from datetime import datetime
from utils.utils import import_attr, get_logger
from typing import Optional


class TensorboardWriter:
    """
    A wrapper class for Tensorboard SummaryWriter.
    """

    def __init__(
        self,
        log_dir: str | pathlib.Path,
        logger: Optional[logging.Logger] = None,
        avail_modules: list[str] = ["torch.utils.tensorboard", "tensorboardX"],
    ):
        """
        Parameters
        ----------
        log_dir : str | pathlib.Path
            Directory to save Tensorboard logs
        logger : logging.Logger
            Logger for logging messages, if None, a default logger will be created
        avail_modules : list of str
            List of module names to try importing SummaryWriter from. Default includes 'torch.utils.tensorboard' and 'tensorboardX'.
        """

        self.logger = logger if logger is not None else get_logger("TensorboardWriter")
        try:
            self.writer, self.selected_module = import_attr(
                "SummaryWriter", avail_modules, str(log_dir)
            )
        except ImportError as e:
            self.logger.error(str(e))
            self.logger.warning(
                "Tensorboard is not installed. Tensorboard logging is disabled."
            )
            self.writer = None
            self.selected_module = "None"

        self.step = 0
        self.mode = ""

        self.tb_writer_ftns = {
            "add_scalar",
            "add_scalars",
            "add_image",
            "add_images",
            "add_audio",
            "add_text",
            "add_histogram",
            "add_pr_curve",
            "add_embedding",
        }
        self.tag_mode_exceptions = {"add_histogram", "add_embedding"}
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        """
        Sets the step for the next visualization
        Parameters
        ----------
        step : int
            Step value to record
        mode : str
            The mode of the current step, e.g., 'train' or 'valid'
        """
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            secs = duration.total_seconds()
            if secs > 0:
                self.add_scalar("steps_per_sec", 1 / secs)
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        Parameters
        ----------
        name : str
            Name of the SummaryWriter method to call
        Returns
        -------
            add_data() methods of tensorboard with additional information (step, tag) added. if `name` not found in `self.tb_writer_ftns`, return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = "{}/{}".format(tag, self.mode)
                    add_data(tag, data, global_step=self.step, *args, **kwargs)

            return wrapper

        if hasattr(self.writer, name):
            return getattr(self.writer, name)

        raise AttributeError(
            f"type object '{self.selected_module}' has no attribute '{name}'"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.flush()
        self.close()
        return False

    def flush(self):
        if self.writer is not None:
            self.logger.info("Flushing Tensorboard writer...")
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.logger.info("Closing Tensorboard writer...")
            self.writer.close()
