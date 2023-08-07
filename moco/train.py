import logging
import os
from argparse import ArgumentParser

import torch.cuda

from moco.cfg import load_config, Config
from moco.constants import LOG_FILE_NAME
from moco.data.data import get_loaders
from moco.model.moco import MoCo
from moco.optimisation import get_optimiser
from moco.persistence import restore_from_checkpoint
from moco.trainer.trainer import Trainer
from moco.utl import mkdir, cosine_scheduler


def train_moco(config: Config):
    if torch.cuda.is_available() and config.optim.device == 'cpu':
        logging.warning('CUDA is available, but chosen to run on CPU.')

    train_loader, val_loader = get_loaders(config)

    model = MoCo(config).to(config.optim.device)

    optimiser = get_optimiser(model.parameters(), config)

    lr_schedule = cosine_scheduler(
        config.optim.lr, 0, config.optim.epochs, len(train_loader), config.optim.warmup_epochs
    )

    to_restore = restore_from_checkpoint(
        config, model, optimiser
    )
    start_epoch = to_restore['epoch']

    trainer = Trainer(model, optimiser, lr_schedule, config)

    trainer.train(train_loader, val_loader, start_epoch)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--config_path', type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config_path)

    # Set up logging
    mkdir(config.general.output_dir)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(module)s:%(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.general.output_dir, LOG_FILE_NAME)),
            logging.StreamHandler()
        ]
    )

    train_moco(config)


if __name__ == '__main__':
    main()
