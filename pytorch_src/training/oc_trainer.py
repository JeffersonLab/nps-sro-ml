import logging
import torch
from base.trainer import BaseTrainer
from base.model import BaseModel
from base.dataloader import BaseDataLoader
from base.scaler import BaseScaler
from typing import Optional
from utils.graph import find_connected_components_undirected
from models.oc_loss import oc_loss_per_batch


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
        else:
            raise ValueError(
                "nb_keep is zero, no background nodes will be kept. This should never happen."
            )

    return mask


class ObjectCondensationTrainer(BaseTrainer):
    """
    Trainer class for Object Condensation model training.
    """

    def __init__(
        self,
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        config: dict,
        device: torch.device,
        dataloader: BaseDataLoader,
        scaler: Optional[BaseScaler] = None,
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

        # signal normalization
        self.scaler = scaler
        if scaler is not None:
            self.logger.info("Fitting scaler on training data...")
            self.scaler.fit(
                torch.cat([data.x for data in self.dataloader.dataset], dim=0)
            )
            self.scaler.to(self.device)
            self.logger.info("Scaler fitted.")

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        total = len(self.dataloader)
        return base.format(batch_idx, total, 100.0 * batch_idx / total)

    def _train_epoch(self, epoch):

        self.model.train()
        total_loss = 0.0

        for batch_idx, data in enumerate(self.dataloader):

            self.optimizer.zero_grad()
            data = data.to(self.device)

            x = self.scaler(data.x) if self.scaler is not None else data.x
            pos = data.pos
            edge_index = data.edge_index
            batch = data.batch if hasattr(data, 'batch') else None

            components = find_connected_components_undirected(x.size(0), edge_index)
            object_ids = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
            for cid, nodes in enumerate(components):
                object_ids[nodes] = cid + 1  # reserve 0 for background

            mask_scale = self.config.get("mask_scale", None)
            if mask_scale is not None:
                mask = create_sample_mask(object_ids, batch=batch, scale=mask_scale)
            else:
                mask = torch.ones(x.size(0), dtype=torch.bool, device=self.device)
            x_c, beta = self.model(x, pos, batch=batch, mask=mask)

            q_min = self.config.get("q_min", 0.3)
            noise_idx = self.config.get("noise_idx", 0)  # reserve 0 for background
            margin = self.config.get("margin", 1.0)

            attr_scale = self.config.get("attr_scale", 1.0)
            repul_scale = self.config.get("repul_scale", 1.0)
            coward_scale = self.config.get("coward_scale", 1.0)
            noise_scale = self.config.get("noise_scale", 0.2)

            l_attr, l_repul, l_coward, l_noise = oc_loss_per_batch(
                x=x_c,
                beta=beta.flatten(),
                object_id=object_ids[mask],
                batch=batch[mask] if batch is not None else None,
                q_min=q_min,
                noise_idx=noise_idx,
                margin=margin,
            )

            l_attr = l_attr * attr_scale
            l_repul = l_repul * repul_scale
            l_coward = l_coward * coward_scale
            l_noise = l_noise * noise_scale

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

        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_dataloader):
                data = data.to(self.device)
                x = self.scaler(data.x) if self.scaler is not None else data.x
                pos = data.pos
                edge_index = data.edge_index
                batch = data.batch if hasattr(data, 'batch') else None

                components = find_connected_components_undirected(x.size(0), edge_index)
                object_ids = torch.zeros(
                    x.size(0), dtype=torch.long, device=self.device
                )
                for cid, nodes in enumerate(components):
                    object_ids[nodes] = cid + 1  # reserve 0 for background

                # for debug purposes
                mask_scale = self.config.get("mask_scale", None)
                if mask_scale is not None:
                    mask = create_sample_mask(object_ids, batch=batch, scale=mask_scale)
                else:
                    mask = torch.ones(x.size(0), dtype=torch.bool, device=self.device)
                x_c, beta = self.model(x, pos, batch=batch, mask=mask)

                q_min = self.config.get("q_min", 0.3)
                noise_idx = self.config.get("noise_idx", 0)  # reserve 0 for background
                margin = self.config.get("margin", 1.0)

                attr_scale = self.config.get("attr_scale", 1.0)
                repul_scale = self.config.get("repul_scale", 1.0)
                coward_scale = self.config.get("coward_scale", 1.0)
                noise_scale = self.config.get("noise_scale", 0.2)

                l_attr, l_repul, l_coward, l_noise = oc_loss_per_batch(
                    x=x_c,
                    beta=beta.flatten(),
                    object_id=object_ids[mask],
                    batch=batch[mask] if batch is not None else None,
                    q_min=q_min,
                    noise_idx=noise_idx,
                    margin=margin,
                )

                l_attr = l_attr * attr_scale
                l_repul = l_repul * repul_scale
                l_coward = l_coward * coward_scale
                l_noise = l_noise * noise_scale
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
