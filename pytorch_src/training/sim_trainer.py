import logging
import torch
from base.trainer import BaseTrainer
from base.model import BaseModel
from base.dataloader import BaseDataLoader
from base.scaler import BaseScaler
from typing import Optional
from models.oc_loss import oc_loss_per_batch
from utils.graph import create_unique_object_ids


class ObjectCondensationTrainer(BaseTrainer):
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
            y = data.y.squeeze(-1).long()
            pos = data.pos
            batch = getattr(data, "batch", None)

            object_ids = create_unique_object_ids(
                y, batch, noise_idx=self.config.get("noise_idx", -1)
            )

            x_c, beta = self.model(x, pos, batch=batch)
            beta = beta.squeeze(-1)

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

            # self.writer.add_scalar('x_c_max', x_c.max().item())
            # self.writer.add_scalar('x_c_mean', x_c.mean().item())
            # self.writer.add_scalar('x_c_std', x_c.std().item())
            # self.writer.add_scalar('beta_mean', beta.mean().item())
            # self.writer.add_scalar('beta_std', beta.std().item())

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
                y = data.y.squeeze(-1).long()
                batch = getattr(data, "batch", None)

                object_ids = create_unique_object_ids(
                    y, batch, noise_idx=self.config.get("noise_idx", -1)
                )
                x_c, beta = self.model(x, pos, batch=batch)
                beta = beta.squeeze(-1)

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

                # self.writer.add_scalar('x_c_max', x_c.max().item())
                # self.writer.add_scalar('x_c_mean', x_c.mean().item())
                # self.writer.add_scalar('x_c_std', x_c.std().item())
                # self.writer.add_scalar('beta_mean', beta.mean().item())
                # self.writer.add_scalar('beta_std', beta.std().item())

                if batch_idx % 10 == 0:
                    self.writer.add_histogram("beta_valid", beta, bins='auto')

                total_loss += loss.item()

        self.writer.add_scalar('total_loss', total_loss)

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            "loss": total_loss,
        }
