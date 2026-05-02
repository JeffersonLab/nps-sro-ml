import logging
from typing import Optional

import torch
import torch.nn.functional as F

from base.dataloader import BaseDataLoader
from base.model import BaseModel
from base.trainer import BaseTrainer
from models.oc_loss import oc_loss_per_batch
from training.oc_trainer import create_sample_mask
from utils.graph import (
    create_unique_object_ids,
    pack_to_graph_batches,
    reorder_from_graph_batches,
)


class MultiPulseOCTrainer(BaseTrainer):
    """
    Trainer for the multi-pulse OC architecture.

    Object labels are read directly from `data.y`. For overlapping clusters,
    `data.y` should provide one or more object IDs per node, padded with
    `noise_idx` for unused slots.
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

        self.model.configure_input_preprocessing(self.config)

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        total = len(self.dataloader)
        return base.format(batch_idx, total, 100.0 * batch_idx / total)

    def _normalize_y_object_ids(
        self,
        y: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise_idx = self.config.get("noise_idx", -1)
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        elif y.ndim == 2 and y.shape[-1] == 1:
            pass
        elif y.ndim > 2:
            y = y.reshape(y.shape[0], -1)

        y = y.long()
        token_object_ids = torch.full_like(y, noise_idx)
        for slot in range(y.shape[1]):
            token_object_ids[:, slot] = create_unique_object_ids(
                y[:, slot], batch, noise_idx=noise_idx
            )

        node_object_ids = torch.full(
            (y.shape[0],),
            noise_idx,
            dtype=torch.long,
            device=y.device,
        )
        signal_mask = token_object_ids != noise_idx
        has_signal = signal_mask.any(dim=-1)
        first_signal_idx = signal_mask.long().argmax(dim=-1)
        node_object_ids[has_signal] = token_object_ids[
            has_signal, first_signal_idx[has_signal]
        ]
        return node_object_ids, token_object_ids

    def _apply_downsampling(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        object_ids: torch.Tensor,
        extra_tensors: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        mask_scale = self.config.get("mask_scale", None)
        noise_idx = self.config.get("noise_idx", -1)
        extras = {} if extra_tensors is None else dict(extra_tensors)

        if mask_scale is None:
            return x, pos, batch, object_ids, extras

        mask = create_sample_mask(
            object_ids,
            batch=batch,
            scale=mask_scale,
            bkg_id=noise_idx,
        )

        x = x[mask]
        pos = pos[mask]
        object_ids = object_ids[mask]
        batch = batch[mask]
        for key, value in extras.items():
            extras[key] = value[mask]

        return x, pos, batch, object_ids, extras

    def _pack_optional_graph_tensor(
        self,
        tensor: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        packed, _, _ = pack_to_graph_batches(tensor, [], batch=batch)
        return packed[0]

    def _align_token_target_width(
        self,
        tensor: torch.Tensor,
        target_width: int,
        pad_value: int,
    ) -> torch.Tensor:
        width = tensor.shape[-1]
        if width == target_width:
            return tensor
        if width > target_width:
            raise ValueError(
                "Target object-id width exceeds model pulse-token width. "
                f"Got {width} target slots but only {target_width} pulse tokens. "
                "Increase `num_pulse_tokens` or set `max_objects_per_block` to match the dataset."
            )

        pad_shape = list(tensor.shape)
        pad_shape[-1] = target_width - width
        pad = torch.full(
            pad_shape,
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, pad], dim=-1)

    def _pool_node_outputs(
        self,
        cluster_outputs: dict[str, torch.Tensor],
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        refined_score = cluster_outputs["refined_pulse_score"]
        cluster_beta = cluster_outputs["cluster_seedness_beta"]
        cluster_z = cluster_outputs["latent_cluster_coordinate_z"]
        cluster_mask = cluster_outputs["cluster_token_mask"]

        token_weight = refined_score * cluster_mask.to(refined_score.dtype)
        token_weight = token_weight / token_weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        x_c = (cluster_z * token_weight.unsqueeze(-1)).sum(dim=2)
        beta = cluster_beta.max(dim=-1, keepdim=True).values

        x_c = x_c * node_mask.unsqueeze(-1).to(x_c.dtype)
        beta = beta * node_mask.unsqueeze(-1).to(beta.dtype)
        return x_c, beta

    def _compute_oc_losses(
        self,
        x_c: torch.Tensor,
        beta: torch.Tensor,
        object_ids: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        return (
            l_attr * attr_scale,
            l_repul * repul_scale,
            l_coward * coward_scale,
            l_noise * noise_scale,
        )

    def _compute_proposal_score_loss(
        self,
        pulse_score: torch.Tensor,
        token_mask: torch.Tensor,
        signal_token_targets: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = token_mask
        if valid_mask.sum() == 0:
            return pulse_score.new_zeros(())
        return F.binary_cross_entropy(
            pulse_score[valid_mask],
            signal_token_targets[valid_mask],
        )

    def _compute_refined_score_loss(
        self,
        refined_score: torch.Tensor,
        token_mask: torch.Tensor,
        signal_token_targets: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = token_mask
        if valid_mask.sum() == 0:
            return refined_score.new_zeros(())
        return F.binary_cross_entropy(
            refined_score[valid_mask],
            signal_token_targets[valid_mask],
        )

    def _compute_sparsity_loss(
        self,
        pulse_score: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_scores = pulse_score[token_mask]
        if valid_scores.numel() == 0:
            return pulse_score.new_zeros(())
        return valid_scores.mean()

    def _compute_optional_regression_losses(
        self,
        proposal: dict[str, torch.Tensor],
        cluster_outputs: dict[str, torch.Tensor],
        optional_targets: dict[str, Optional[torch.Tensor]],
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = proposal["pulse_score"].new_zeros(())
        pulse_mask = optional_targets.get("pulse_mask")
        pulse_time = optional_targets.get("pulse_time")
        pulse_amp = optional_targets.get("pulse_amplitude")
        pulse_width = optional_targets.get("pulse_width")

        if pulse_mask is None:
            return zero, zero, zero

        pulse_mask = pulse_mask.to(dtype=torch.bool) & token_mask

        time_loss = zero
        if pulse_time is not None:
            time_loss = F.smooth_l1_loss(
                proposal["pulse_time"][pulse_mask],
                pulse_time[pulse_mask].to(proposal["pulse_time"].dtype),
            )

        amp_loss = zero
        if pulse_amp is not None:
            amp_loss = F.smooth_l1_loss(
                cluster_outputs["refined_charge"][pulse_mask],
                pulse_amp[pulse_mask].to(cluster_outputs["refined_charge"].dtype),
            )

        width_loss = zero
        if pulse_width is not None:
            width_loss = F.smooth_l1_loss(
                proposal["pulse_width"][pulse_mask],
                pulse_width[pulse_mask].to(proposal["pulse_width"].dtype),
            )

        return time_loss, amp_loss, width_loss

    def _compute_token_oc_loss(
        self,
        cluster_outputs: dict[str, torch.Tensor],
        token_object_ids_graph: torch.Tensor,
        batch_graph: torch.Tensor,
        token_mask: torch.Tensor,
        optional_targets: dict[str, Optional[torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        token_object_ids = token_object_ids_graph
        pulse_object_id = optional_targets.get("pulse_object_id")
        if pulse_object_id is not None:
            token_object_ids = pulse_object_id.to(token_object_ids.dtype).long()

        token_batch = batch_graph.unsqueeze(-1).expand_as(token_object_ids)

        flat_mask = token_mask.reshape(-1)
        flat_z = cluster_outputs["latent_cluster_coordinate_z"].reshape(
            -1, cluster_outputs["latent_cluster_coordinate_z"].shape[-1]
        )
        flat_beta = cluster_outputs["cluster_seedness_beta"].reshape(-1)
        flat_obj = token_object_ids.reshape(-1)
        flat_batch = token_batch.reshape(-1)
        if flat_mask.sum() == 0:
            zero = flat_beta.new_zeros(())
            return zero, zero, zero, zero

        return self._compute_oc_losses(
            x_c=flat_z[flat_mask],
            beta=flat_beta[flat_mask],
            object_ids=flat_obj[flat_mask],
            batch=flat_batch[flat_mask],
        )

    def _forward_losses(self, data):
        score_scale = self.config.get("proposal_score_scale", 5.0)
        refined_score_scale = self.config.get("refined_score_scale", 1.0)
        sparsity_scale = self.config.get("sparsity_scale", 0.01)
        time_scale = self.config.get("time_scale", 0.0)
        amp_scale = self.config.get("amp_scale", 0.0)
        width_scale = self.config.get("width_scale", 0.0)

        x = self.model.preprocess_features(data)
        pos = data.pos
        batch = self.model.get_batch_vector(x, getattr(data, "batch", None))
        y = data.y
        object_ids, token_object_ids = self._normalize_y_object_ids(y, batch)

        extra_tensors = {}
        for key in (
            "pulse_time",
            "pulse_amplitude",
            "pulse_width",
            "pulse_mask",
            "pulse_object_id",
        ):
            if hasattr(data, key):
                extra_tensors[key] = getattr(data, key)
        extra_tensors["token_object_id"] = token_object_ids

        x, pos, batch, object_ids, extra_tensors = self._apply_downsampling(
            x, pos, batch, object_ids, extra_tensors=extra_tensors
        )

        x_graph, pos_graph, fea_mask, node_mask, idx_out = self.model.prepare_graph_inputs(
            x, pos, batch
        )
        packed_graphs, _, _ = pack_to_graph_batches(
            batch.unsqueeze(-1).float(),
            [object_ids.unsqueeze(-1).float()],
            batch=batch,
        )
        batch_graph = packed_graphs[0].squeeze(-1).long()
        object_ids_graph = packed_graphs[1].squeeze(-1).long()

        optional_targets: dict[str, Optional[torch.Tensor]] = {}
        for key, value in extra_tensors.items():
            optional_targets[key] = self._pack_optional_graph_tensor(value, batch)

        proposal = self.model.propose_pulses(x_graph, pos_graph, fea_mask, node_mask)
        cluster_outputs = self.model.cluster_pulses(
            proposal=proposal,
            prune_mask=None,
            soft_pruning=True,
        )

        token_mask = proposal["token_mask"] & node_mask.unsqueeze(-1)
        token_object_ids_graph = self._align_token_target_width(
            optional_targets["token_object_id"].long(),
            proposal["pulse_score"].shape[-1],
            self.config.get("noise_idx", -1),
        )
        signal_token_targets = token_object_ids_graph != self.config.get("noise_idx", -1)
        signal_token_targets = signal_token_targets & token_mask
        signal_token_targets = signal_token_targets.to(proposal["pulse_score"].dtype)
        proposal_score_loss = self._compute_proposal_score_loss(
            proposal["pulse_score"], token_mask, signal_token_targets
        )
        refined_score_loss = self._compute_refined_score_loss(
            cluster_outputs["refined_pulse_score"],
            token_mask,
            signal_token_targets,
        )
        sparsity_loss = self._compute_sparsity_loss(
            proposal["pulse_score"], token_mask
        )
        time_loss, amp_loss, width_loss = self._compute_optional_regression_losses(
            proposal, cluster_outputs, optional_targets, token_mask
        )
        l_attr, l_repul, l_coward, l_noise = self._compute_token_oc_loss(
            cluster_outputs,
            token_object_ids_graph,
            batch_graph,
            token_mask,
            optional_targets,
        )

        loss = (
            l_attr
            + l_repul
            + l_coward
            + l_noise
            + score_scale * proposal_score_loss
            + refined_score_scale * refined_score_loss
            + sparsity_scale * sparsity_loss
            + time_scale * time_loss
            + amp_scale * amp_loss
            + width_scale * width_loss
        )

        pooled_x_c, pooled_beta = self._pool_node_outputs(cluster_outputs, node_mask)
        pooled_x_c = reorder_from_graph_batches(pooled_x_c, idx_out)
        pooled_beta = reorder_from_graph_batches(pooled_beta, idx_out).squeeze(-1)

        return {
            "loss": loss,
            "l_attr": l_attr,
            "l_repul": l_repul,
            "l_coward": l_coward,
            "l_noise": l_noise,
            "proposal_score_loss": proposal_score_loss,
            "refined_score_loss": refined_score_loss,
            "sparsity_loss": sparsity_loss,
            "time_loss": time_loss,
            "amp_loss": amp_loss,
            "width_loss": width_loss,
            "beta": pooled_beta,
            "x_c": pooled_x_c,
        }

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        for batch_idx, data in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            data = data.to(self.device)

            outputs = self._forward_losses(data)
            outputs["loss"].backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.dataloader) + batch_idx)
            for key in (
                "loss",
                "l_attr",
                "l_repul",
                "l_coward",
                "l_noise",
                "proposal_score_loss",
                "refined_score_loss",
                "sparsity_loss",
                "time_loss",
                "amp_loss",
                "width_loss",
            ):
                self.writer.add_scalar(key, outputs[key].item())

            total_loss += outputs["loss"].item()

            if batch_idx % 10 == 0:
                self.logger.info(
                    'Train Epoch: {} {} Loss: {:.6f}'.format(
                        epoch, self._progress(batch_idx), outputs["loss"].item()
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
                outputs = self._forward_losses(data)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_dataloader) + batch_idx, 'valid'
                )
                for key in (
                    "loss",
                    "l_attr",
                    "l_repul",
                    "l_coward",
                    "l_noise",
                    "proposal_score_loss",
                    "refined_score_loss",
                    "sparsity_loss",
                    "time_loss",
                    "amp_loss",
                    "width_loss",
                ):
                    self.writer.add_scalar(key, outputs[key].item())

                total_loss += outputs["loss"].item()

        self.writer.add_scalar('total_loss', total_loss)

        return {"loss": total_loss}
