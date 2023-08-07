import logging
import os
from dataclasses import asdict

import numpy as np
import wandb
import torch
from lightly.loss import NTXentLoss
from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule
from torch.optim import Optimizer

from moco.cfg import Config
from moco.constants import LATEST_MODEL_FILE_NAME
from moco.model.moco import MoCo
from moco.trainer.metrics import contrastive_accuracy


class Trainer:
    def __init__(self, model: MoCo, optimiser: Optimizer, lr_schedule, config: Config):
        self.model = model
        self.optimiser = optimiser
        self.lr_schedule = lr_schedule
        self.config = config

        self.criterion = NTXentLoss(memory_bank_size=config.model.memory_bank_size)

        if config.general.log_to_wandb:
            wandb.init(
                project='phd', config=asdict(config),
                name=f'MoCo : {config.data.dataset}',
                tags=[f'run_id: {config.general.run_id}'],
            )

    def train_one_epoch(self, train_loader, epoch):
        train_loss = 0.0
        train_contrastive_accuracy = 0.0

        momentum_val = cosine_schedule(epoch, self.config.optim.epochs, 0.996, 1)
        for batch_num, batch in enumerate(train_loader):
            global_iteration = len(train_loader) * epoch + batch_num

            self.optimiser.param_groups[0]['lr'] = self.lr_schedule[global_iteration]

            x_query, x_key = batch[0]

            update_momentum(self.model.backbone, self.model.backbone_momentum, m=momentum_val)
            update_momentum(
                self.model.projection_head, self.model.projection_head_momentum, m=momentum_val
            )

            x_query = x_query.to(self.config.optim.device)
            x_key = x_key.to(self.config.optim.device)

            query = self.model(x_query)
            key = self.model.forward_momentum(x_key)

            loss = self.criterion(query, key)

            train_loss += float(loss.item())
            train_contrastive_accuracy += contrastive_accuracy(query, key, self.config)

            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()

        train_loss /= len(train_loader)
        train_contrastive_accuracy /= len(train_loader)

        return {
            'train/loss': train_loss,
            'val/contrastive_accuracy': train_contrastive_accuracy
        }

    def validate_one_epoch(self, val_loader):
        self.model.eval()

        val_loss = 0.0
        val_contrastive_accuracy = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_query, x_key = batch[0]

                x_query = x_query.to(self.config.optim.device)
                x_key = x_key.to(self.config.optim.device)

                query = self.model(x_query)
                key = self.model.forward_momentum(x_key)

                loss = self.criterion(query, key)

                val_loss += float(loss.item())

                val_contrastive_accuracy += contrastive_accuracy(query, key, self.config)

        self.model.train()

        val_loss /= len(val_loader)
        val_contrastive_accuracy /= len(val_loader)

        return {
            'val/loss': val_loss,
            'val/contrastive_accuracy': val_contrastive_accuracy
        }

    def train(self, train_loader, val_loader, start_epoch):
        best_val_loss = np.inf

        for epoch in range(start_epoch, self.config.optim.epochs):
            logging.info('Epoch %s/%s', epoch + 1, self.config.optim.epochs)

            train_metrics = self.train_one_epoch(train_loader, epoch)

            val_metrics = self.validate_one_epoch(val_loader)

            if self.config.general.log_to_wandb:
                wandb.log({**train_metrics, **val_metrics})

            val_loss = val_metrics['val/loss']

            state_dict = {
                'model': self.model.state_dict(),
                'optimiser': self.optimiser.state_dict(),
                'epoch': epoch + 1,
            }

            torch.save(
                state_dict,
                os.path.join(self.config.general.output_dir, LATEST_MODEL_FILE_NAME)
            )

            if epoch % self.config.general.checkpoint_freq == 0 or (epoch + 1) == self.config.optim.epochs:
                torch.save(
                    state_dict,
                    os.path.join(self.config.general.output_dir, f'simclr-epoch-{epoch}.pth')
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    state_dict,
                    os.path.join(self.config.general.output_dir, 'best.pth')
                )
