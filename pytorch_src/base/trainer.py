import pathlib
import torch
import logging
import numpy as np

from utils.tensorboard import TensorboardWriter
from utils.utils import get_logger
from abc import ABC, abstractmethod
from typing import Optional


class BaseTrainer(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: dict,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.logger = logger if logger is not None else get_logger("trainer")

        self.epochs = config.get("epochs", 5)
        self.mnt_mode = config.get("mnt_mode", "min")
        self.mnt_metric = config.get("mnt_metric", "loss")
        self.save_dir = config.get("save_dir", "./saved")
        self.log_dir = config.get("log_dir", "./logs")

        assert self.mnt_mode in ["min", "max"]

        self.checkpoint_dir = pathlib.Path(self.save_dir).resolve()
        self.log_dir = pathlib.Path(self.log_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.save_period = config.get("save_period", 1)
        self.mnt_best = np.inf if self.mnt_mode == "min" else -np.inf

        self.early_stop = config.get("early_stop", None)
        if self.early_stop is None or self.early_stop <= 0:
            self.early_stop = float("inf")

        self.start_epoch = 1
        self.writer = TensorboardWriter(self.log_dir, self.logger)

    @abstractmethod
    def _train_epoch(self, epoch) -> dict:
        """
        Parameters
        ----------
        epoch : int
            Current epoch number
        Returns
        -------
        log : dict
            Dictionary containing average loss and metric values for the epoch
        """
        raise NotImplementedError

    def train(self) -> "BaseTrainer":
        """
        Full training logic
        Returns
        -------
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            log = {"epoch": epoch}
            log.update(result)
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            if self.mnt_metric not in log:
                raise KeyError(
                    f"Metric '{self.mnt_metric}' not returned by _train_epoch"
                )

            best = False
            improved = (
                self.mnt_mode == "min" and log[self.mnt_metric] <= self.mnt_best
            ) or (self.mnt_mode == "max" and log[self.mnt_metric] >= self.mnt_best)

            if improved:
                self.mnt_best = log[self.mnt_metric]
                not_improved_count = 0
                best = True

            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                self.logger.info(
                    "Validation performance didn\'t improve for {} epochs. "
                    "Training stops.".format(self.early_stop)
                )
                break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

        return self

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            "arch": type(self.model).__name__,
            "module": self.model.__class__.__module__,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
            "metadata": self._prepare_model_metadata(),
        }

        filename = self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth"
        self.logger.info(f"Saving checkpoint: {filename} ...")
        torch.save(state, filename)

        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            self.logger.info(f"Saving current best: {best_path} ...")
            torch.save(state, best_path)

    def _resume_checkpoint(self, resume_path: str | pathlib.Path):
        raise NotImplementedError

    def _prepare_model_metadata(self) -> dict:
        """
        Return a dictionary of model attributes to save via torch.save. Override this method in your subclass if you need to save more attributes to reconstruct the model later.
        """
        return {
            k: v
            for k, v in self.model.__dict__.items()
            if not k.startswith("_")  # Exclude private/internal attributes
            and not isinstance(v, torch.nn.Module)  # Exclude nested modules
            and not isinstance(v, torch.optim.Optimizer)  # Exclude optimizers
            and not isinstance(
                v, torch.Tensor
            )  # Exclude tensors (state_dict handles them)
            and not callable(v)  # Exclude functions/methods
        }
