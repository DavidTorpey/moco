import logging
import os
from pathlib import Path

import numpy as np

from moco.cfg import Config
from moco.data.dataset import MoCoDataset


def get_el(config: Config, train_aug, val_aug):
    root = os.path.join(config.data.dataset_root, 'dataset', 'all_data')
    all_paths = np.array(list(Path(root).rglob('*.png')) + list(Path(root).rglob('*.jpg')))
    np.random.shuffle(all_paths)

    if config.data.max_images is not None:
        all_paths = all_paths[:config.data.max_images]

    n_train = int(len(all_paths) * 0.8)

    train_paths = all_paths[:n_train]
    val_paths = all_paths[n_train:]

    train_dataset = MoCoDataset(train_paths, train_aug)

    val_dataset = MoCoDataset(val_paths, val_aug)

    logging.info('Initialised EL dataset: Train=%s, Val=%s', len(train_dataset), len(val_dataset))

    return train_dataset, val_dataset
