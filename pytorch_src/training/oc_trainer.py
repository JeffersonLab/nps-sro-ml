import logging
import torch
import pathlib
import pandas as pd
from typing import Any, Optional, Tuple, Dict
from torch.export import Dim

from base.trainer import BaseTrainer
from base.model import BaseModel
from base.dataloader import BaseDataLoader
from datasets.nps import get_node_index_from_position
from models.oc_loss import oc_loss_per_batch
from utils.graph import (
    create_unique_object_ids,
    pack_to_graph_batches,
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
        pass

    def _unpack_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unpack features to desired shape. This is a placeholder method that should be implemented by subclasses to handle specific feature formats. For example, if the input features are packed as (E, t) pairs for pulses, this method can reshape them and create a mask for valid pulses.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape [..., D] where D is the feature dimension.

        Returns
        -------
        x_out : torch.Tensor
        mask : torch.Tensor
        """
        return x, torch.ones(x.shape, dtype=torch.bool, device=x.device)

    def _preprocess_features(self, data) -> torch.Tensor:
        """
        Pre-process features as needed before feeding into the model. This can include operations like normalization, scaling, or other transformations. By default, it returns the input features unchanged, but subclasses can override this method to implement specific pre-processing steps.

        Parameters
        ----------
        data : Any
            The input data object containing features and possibly other information needed for pre-processing.

        Returns
        -------
        torch.Tensor
            Pre-processed features, shape [..., D'] where D' may be different from D depending on the transformations applied.
        """
        return data.x

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

    def _prepare_graph_inputs(
        self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare graph inputs by packing features into graph batches and unpacking the feature vector into the desired shape. This is a helper function that can be used in both training and validation steps to ensure the input features are in the correct format for the model.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape [N, D].
        pos : torch.Tensor
            Node positions, shape [N, pos_dim].
        batch : torch.Tensor
            Tensor of shape [N] indicating the graph index for each node in a batched setting.

        Return
        ------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing
            - x: Packed and unpacked features, shape [G, N_max, ...] where the remaining dimensions depend on the output of _unpack_features.
            - pos: Packed positions, shape [G, N_max, pos_dim].
            - fea_mask: Mask for valid features, shape [G, N_max, ...].
            - node_mask: Mask for valid nodes, shape [G, N_max].
            - idx_out: Indices to reorder outputs back to original node order, shape [N].
        """
        # nodes, D -> G, N_max, D
        # valid = [G, N_max] (True for valid node, False for padded)
        x_out, idx_out, node_mask = pack_to_graph_batches(x, [pos], batch=batch)
        x = x_out[0]  # [G, N_max, D]
        pos = x_out[1]  # [G, N_max, pos_dim]
        # [G, N_max, P_max, 2], [G, N_max, P_max]
        x, fea_mask = self._unpack_features(x)
        return x, pos, fea_mask, node_mask, idx_out

    def _get_batch_vector(
        self, x: torch.Tensor, batch: Optional[torch.Tensor]
    ) -> torch.LongTensor:
        """
        Return a valid batch vector. Single-graph inputs are treated as batch 0.
        """
        if batch is None:
            return torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return batch

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

            x = self._preprocess_features(data)
            y = data.y.squeeze(-1).long()
            pos = data.pos
            batch = self._get_batch_vector(x, getattr(data, "batch", None))
            object_ids = create_unique_object_ids(y, batch, noise_idx)

            x, pos, batch, object_ids = self._apply_downsampling(
                x, pos, batch, object_ids
            )

            x, pos, fea_mask, node_mask, idx_out = self._prepare_graph_inputs(
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

                x = self._preprocess_features(data)
                pos = data.pos
                y = data.y.squeeze(-1).long()
                batch = self._get_batch_vector(x, getattr(data, "batch", None))
                object_ids = create_unique_object_ids(y, batch, noise_idx)

                x, pos, fea_mask, node_mask, idx_out = self._prepare_graph_inputs(
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
        x = self._preprocess_features(data)
        y = data.y.squeeze(-1).long()
        batch = self._get_batch_vector(x, getattr(data, "batch", None))

        noise_idx = self.config.get("noise_idx", -1)
        object_ids = create_unique_object_ids(y, batch, noise_idx)

        x, pos, batch, object_ids = self._apply_downsampling(x, pos, batch, object_ids)
        x, pos, fea_mask, node_mask, _ = self._prepare_graph_inputs(x, pos, batch)

        batch_size = Dim("batch_size", min=1)
        graph_size = Dim("graph_size", min=1)
        dynamic_shapes = {
            "x": {0: batch_size, 1: graph_size},
            "pos": {0: batch_size, 1: graph_size},
            "fea_mask": {0: batch_size, 1: graph_size},
            "node_mask": {0: batch_size, 1: graph_size},
        }

        artifacts_dir = self.save_dir / "onnx_artifacts"
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

    def _unpack_features(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Disentangle energy and time features, creating a mask for valid pulses. Assumes input x has shape [N, L] where L is even and represents (E, t) pairs for pulses. Returns reshaped features and mask.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape [..., L], where L is max feature length (e.g. max number of pulses per node times 2).

        Returns
        -------
        x_out : torch.Tensor
            Shape [..., P, 2], where P = L // 2.
        mask : torch.Tensor
            Shape [..., P], True where pulse is non-zero.
        """
        L = x.shape[-1]
        P = L // 2
        x_out = x.reshape(*x.shape[:-1], P, 2)
        mask = x_out.abs().sum(dim=-1) > 0  # True if either energy or time is non-zero
        return x_out, mask


class WaveformOCTrainer(BaseObjectCondensationTrainer):
    """
    Trainer for object condensation for waveform data. Assumes input features are waveforms of shape [N, L], where N is number of nodes and L is waveform length. Pre-processes waveforms based on VME configuration, e.g. by subtracting pedestal values.
    """

    def _setup(self):
        """
        Load VME and VTP configuration from CSV files if specified in the config. The configurations are stored as dicts of tensors keyed by column name (excluding 'channel').
        """
        vme_path = self.config.get("vme_config_path", self.config.get("vme_config"))
        vtp_path = self.config.get("vtp_config_path", self.config.get("vtp_config"))
        self.vme_config = self._load_vme_config(vme_path)
        self.vtp_config = self._load_vtp_config(vtp_path)

    def _load_config(self, path: Optional[pathlib.Path]) -> Dict[str, torch.Tensor]:
        """
        Load configuration CSV and return a dict of tensors keyed by column name (excluding 'channel').

        Parameters
        ----------
        path : pathlib.Path or None
            Path to CSV file. If None, returns empty dict.

        Returns
        -------
        Dict[str, torch.Tensor]
            Mapping from config field name to 1D float32 tensor.
        """
        if path is None:
            return {}

        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        df = pd.read_csv(path)

        if "channel" not in df.columns:
            raise ValueError("CSV must contain a 'channel' column")

        config: Dict[str, torch.Tensor] = {}

        for col in df.columns:
            if col == "channel":
                continue

            values = df[col].to_numpy(dtype="float32", copy=False)
            config[col] = torch.from_numpy(values)

        return config

    def _load_vtp_config(self, path: Optional[pathlib.Path]) -> Dict[str, torch.Tensor]:
        """Load VTP configuration from CSV file."""
        return self._load_config(path)

    def _load_vme_config(self, path: Optional[pathlib.Path]) -> Dict[str, torch.Tensor]:
        """Load VME configuration from CSV file."""
        return self._load_config(path)

    def _preprocess_features(self, data: Any) -> torch.Tensor:
        """
        Pre-process waveform features based on VME configuration. For example, subtract pedestal values from waveforms if specified in the VME config.

        Parameters
        ----------
        data : Any
            The input data object with fields 'x' for waveforms and 'pos' for node positions. The 'pos' field is used to determine the channel index for each waveform based on the x and y coordinates.

        Returns
        -------
        torch.Tensor
            Pre-processed waveform tensor of shape [N, L], where N is number of nodes and L is waveform length.
        """
        wf = data.x
        pos = data.pos
        channels = get_node_index_from_position(pos[:, 1], pos[:, 0])
        return self._preprocess_wf(wf, channels)

    def _preprocess_wf(self, wf: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
        """
        Pre-process waveform based on VME configuration. Subtract pedestal if specified.

        Parameters
        ----------
        wf: torch.Tensor
            Input waveform tensor of shape [N, L], where N is number of nodes, L is waveform length.

        channels : torch.Tensor
            Tensor of shape [N,] indicating the channel index for each waveform.

        Returns
        -------
        torch.Tensor
            Processed waveform tensor of shape [N, L].
        """
        if self.vme_config is None:
            return wf

        ped = self.vme_config.get("FADC250_ALLCH_PED")
        if ped is None:
            return wf

        if ped.ndim == 1:
            ped = ped.unsqueeze(-1)  # match wf shape if needed

        if channels.shape[0] != wf.shape[0]:
            raise ValueError(
                f"Channels tensor length ({channels.shape[0]}) does not match number of waveforms ({wf.shape[0]})."
            )

        if channels.numel() > 0 and (
            channels.min() < 0 or channels.max() >= ped.shape[0]
        ):
            raise ValueError(
                f"Waveform channels must be in [0, {ped.shape[0]}), got "
                f"[{channels.min().item()}, {channels.max().item()}]."
            )

        wf = wf - ped.to(wf.device)[channels.long()]
        return wf
