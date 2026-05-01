import logging
import torch
import pathlib
from typing import Optional, Tuple
from torch.export import Dim

from base.trainer import BaseTrainer
from base.model import BaseModel
from base.dataloader import BaseDataLoader
from models.oc_loss import oc_loss_per_batch
from utils.graph import (
    create_unique_object_ids,
    reorder_from_graph_batches,
)


def create_sample_mask(
    object_ids: torch.Tensor,
    batch: Optional[torch.LongTensor] = None,
    scale: float = 5.0,
    bkg_id: int = 0,
) -> torch.Tensor:
    """
    Create a mask to only retain a subset of background nodes for training.

    Parameters
    ----------
    object_ids: torch.Tensor
        Tensor of shape (N,) containing object IDs for each node. Background nodes should have the ID equal to bkg_id.
    batch: Optional[torch.LongTensor]
        Tensor of shape (N,) indicating the graph index for each node in a batched setting. If None, all nodes are considered to belong to a single graph.
    scale: float
        Scaling factor to determine how many background nodes to keep relative to the number of signal nodes.
    bkg_id: int
        The object ID that indicates background nodes.

    Returns
    -------
    torch.Tensor
        A boolean mask tensor of shape (N,) where True indicates the node is kept for training.
    """
    device = object_ids.device
    x_size = object_ids.size(0)
    if batch is None:
        batch = torch.zeros(x_size, dtype=torch.long, device=device)

    is_bkg = object_ids == bkg_id
    is_sig = ~is_bkg
    mask = torch.where(is_sig, True, False)

    for b in batch.unique(sorted=True):
        node_mask = batch == b
        nb_sig = (is_sig & node_mask).sum().item()
        bkg_indices = torch.nonzero(is_bkg & node_mask, as_tuple=False).flatten()

        nb_keep = int((nb_sig + 1) * scale)
        nb_keep = min(nb_keep, bkg_indices.numel())

        if nb_keep > 0:
            perm = torch.randperm(bkg_indices.size(0), device=device)
            selected = bkg_indices[perm[:nb_keep]]
            mask[selected] = True

    return mask


class BaseObjectCondensationTrainer(BaseTrainer):
    """
    Base Object Condensation Trainer class. Implements the common training loop and loss computation for object condensation, while allowing for flexible input feature formats and pre-processing steps through methods that can be overridden by subclasses.
    """

    def __init__(
        self,
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        config: dict,
        device: torch.device,
        dataloader: BaseDataLoader,
        valid_dataloader: Optional[BaseDataLoader] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(model, optimizer, config, logger)

        self.dataloader = dataloader
        self.valid_dataloader = valid_dataloader
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.do_validation = self.valid_dataloader is not None

        self._setup()

    def _progress(self, batch_idx):
        """Return a string progress indicator for the current batch index."""
        base = '[{}/{} ({:.0f}%)]'
        total = len(self.dataloader)
        return base.format(batch_idx, total, 100.0 * batch_idx / total)

    def _setup(self):
        """
        Set additonal attributes needed for training, such as loading scalers or other pre-processing tools.
        """
        self.model.configure_input_preprocessing(self.config)

    def _compute_oc_losses(
        self,
        x_c: torch.Tensor,
        beta: torch.Tensor,
        object_ids: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the object condensation losses for a batch of data. This includes the attractive loss, repulsive loss, cowardly loss, and noise loss.

        Parameters
        ----------
        x_c : torch.Tensor
            The condensed coordinates output by the model, shape [N, C].
        beta : torch.Tensor
            The beta values output by the model, shape [N].
        object_ids : torch.Tensor
            Tensor of shape [N] containing the object ID for each node.
        batch : torch.Tensor
            Tensor of shape [N] indicating the graph index for each node in a batched setting.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing the attractive loss, repulsive loss, cowardly loss, and noise loss for the batch.
        """
        q_min = self.config.get("q_min", 0.3)
        noise_idx = self.config.get("noise_idx", -1)
        margin = self.config.get("margin", 1.0)

        attr_scale = self.config.get("attr_scale", 1.0)
        repul_scale = self.config.get("repul_scale", 1.0)
        coward_scale = self.config.get("coward_scale", 1.0)
        noise_scale = self.config.get("noise_scale", 0.0)

        l_attr, l_repul, l_coward, l_noise = oc_loss_per_batch(
            x=x_c,
            beta=beta,
            object_id=object_ids,
            batch=batch,
            q_min=q_min,
            noise_idx=noise_idx,
            margin=margin,
        )

        l_attr *= attr_scale
        l_repul *= repul_scale
        l_coward *= coward_scale
        l_noise *= noise_scale

        return l_attr, l_repul, l_coward, l_noise

    def _apply_downsampling(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        object_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply down sampling to the input tensors. This function should be called right after the data is retrieved from the dataloader, i.e. when the input tensors are still with shape [N,D].

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape [N, D].
        pos : torch.Tensor
            Node positions, shape [N, pos_dim].
        batch : torch.Tensor
            Tensor of shape [N] indicating the graph index for each node in a batched setting.
        object_ids : torch.Tensor
            Tensor of shape [N] containing the object ID for each node.

        Return
        ------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing the downsampled x, pos, batch, and object_ids tensors, all with shape [N', D], [N', pos_dim], [N'], and [N'] respectively, where N' <= N depending on the downsampling applied.
        """
        mask_scale = self.config.get("mask_scale", None)
        noise_idx = self.config.get("noise_idx", -1)

        if mask_scale is None:
            return x, pos, batch, object_ids

        mask = create_sample_mask(
            object_ids,
            batch=batch,
            scale=mask_scale,
            bkg_id=noise_idx,
        )

        x = x[mask]
        pos = pos[mask]
        object_ids = object_ids[mask]

        if batch is not None:
            batch = batch[mask]

        return x, pos, batch, object_ids

    def _train_epoch(self, epoch):

        self.model.train()
        total_loss = 0.0

        noise_idx = self.config.get("noise_idx", -1)
        for batch_idx, data in enumerate(self.dataloader):

            self.optimizer.zero_grad()
            data = data.to(self.device)

            x = self.model.preprocess_features(data)
            y = data.y.squeeze(-1).long()
            pos = data.pos
            batch = self.model.get_batch_vector(x, getattr(data, "batch", None))
            object_ids = create_unique_object_ids(y, batch, noise_idx)

            x, pos, batch, object_ids = self._apply_downsampling(
                x, pos, batch, object_ids
            )

            x, pos, fea_mask, node_mask, idx_out = self.model.prepare_graph_inputs(
                x, pos, batch
            )

            x_c, beta = self.model(x, pos, fea_mask, node_mask)

            x_c = reorder_from_graph_batches(x_c, idx_out)
            beta = reorder_from_graph_batches(beta, idx_out)
            beta = beta.squeeze(-1)

            l_attr, l_repul, l_coward, l_noise = self._compute_oc_losses(
                x_c, beta, object_ids, batch
            )
            loss = l_attr + l_repul + l_coward + l_noise
            loss.backward()

            self.optimizer.step()
            self.writer.set_step((epoch - 1) * len(self.dataloader) + batch_idx)

            self.writer.add_scalar('loss', loss.item())
            self.writer.add_scalar('l_attr', l_attr.item())
            self.writer.add_scalar('l_repul', l_repul.item())
            self.writer.add_scalar('l_coward', l_coward.item())
            self.writer.add_scalar('l_noise', l_noise.item())

            if batch_idx % 10 == 0:
                self.writer.add_histogram("beta_train", beta, bins='auto')

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                self.logger.info(
                    'Train Epoch: {} {} Loss: {:.6f}'.format(
                        epoch, self._progress(batch_idx), loss.item()
                    )
                )

        self.writer.add_scalar("total_loss", total_loss)
        log = {"loss": total_loss}

        if self.lr_scheduler is not None:
            self.writer.add_scalar("lr", self.lr_scheduler.get_last_lr()[0])

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0

        noise_idx = self.config.get("noise_idx", -1)

        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_dataloader):
                data = data.to(self.device)

                x = self.model.preprocess_features(data)
                pos = data.pos
                y = data.y.squeeze(-1).long()
                batch = self.model.get_batch_vector(x, getattr(data, "batch", None))
                object_ids = create_unique_object_ids(y, batch, noise_idx)

                x, pos, fea_mask, node_mask, idx_out = self.model.prepare_graph_inputs(
                    x, pos, batch
                )
                x_c, beta = self.model(x, pos, fea_mask, node_mask)

                x_c = reorder_from_graph_batches(x_c, idx_out)
                beta = reorder_from_graph_batches(beta, idx_out)
                beta = beta.squeeze(-1)

                l_attr, l_repul, l_coward, l_noise = self._compute_oc_losses(
                    x_c, beta, object_ids, batch
                )
                loss = l_attr + l_repul + l_coward + l_noise

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_dataloader) + batch_idx, 'valid'
                )
                self.writer.add_scalar('loss', loss.item())
                self.writer.add_scalar('l_attr', l_attr.item())
                self.writer.add_scalar('l_repul', l_repul.item())
                self.writer.add_scalar('l_coward', l_coward.item())
                self.writer.add_scalar('l_noise', l_noise.item())

                if batch_idx % 10 == 0:
                    self.writer.add_histogram("beta_valid", beta, bins='auto')

                total_loss += loss.item()

        self.writer.add_scalar('total_loss', total_loss)

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            "loss": total_loss,
        }

    def export_onnx(self, pth: str | pathlib.Path = "model_best.onnx"):
        """
        Export the model to ONNX format using a sample batch from the dataloader. The input tensors are prepared in the same way as during training, including packing into graph batches and unpacking features.

        Parameters
        ----------
        pth : str or pathlib.Path
            The path where the ONNX model will be saved.
        """
        self.model.eval()
        data = next(iter(self.dataloader))
        data = data.to(self.device)
        pos = data.pos
        x = self.model.preprocess_features(data)
        y = data.y.squeeze(-1).long()
        batch = self.model.get_batch_vector(x, getattr(data, "batch", None))

        noise_idx = self.config.get("noise_idx", -1)
        object_ids = create_unique_object_ids(y, batch, noise_idx)

        x, pos, batch, object_ids = self._apply_downsampling(x, pos, batch, object_ids)
        x, pos, fea_mask, node_mask, _ = self.model.prepare_graph_inputs(x, pos, batch)

        batch_size = Dim("batch_size", min=1)
        graph_size = Dim("graph_size", min=1)
        dynamic_shapes = {
            "x": {0: batch_size, 1: graph_size},
            "pos": {0: batch_size, 1: graph_size},
            "fea_mask": {0: batch_size, 1: graph_size},
            "node_mask": {0: batch_size, 1: graph_size},
        }

        artifacts_dir = self.checkpoint_dir / "onnx_artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            self.model,
            (x, pos, fea_mask, node_mask),
            str(pth),
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            input_names=["x", "pos", "fea_mask", "node_mask"],
            verify=True,
            report=True,
            artifacts_dir=str(artifacts_dir),
        )


class PulseOCTrainer(BaseObjectCondensationTrainer):
    """
    Trainer for object condensation on simulated data. Assumes input features are packed as (Energy, time) pairs for pulses.
    """

    pass


class WaveformOCTrainer(BaseObjectCondensationTrainer):
    """
    Trainer for object condensation for waveform data. Assumes input features are waveforms of shape [N, L], where N is number of nodes and L is waveform length. Pre-processes waveforms based on VME configuration, e.g. by subtracting pedestal values.
    """

    pass
