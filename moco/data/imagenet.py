import logging
import os
from pathlib import Path

import numpy as np

from moco.cfg import Config
from moco.data.dataset import MoCoDataset


def get_imagenet(config: Config, train_transform, val_transform):
    train_paths = list(map(
        str,
        list(Path(
            os.path.join(config.data.dataset_root, 'imagenet')
        ).rglob('*.JPEG'))
    ))

    train_paths = np.array(train_paths)
    np.random.shuffle(train_paths)
    if config.data.max_images is not None:
        train_paths = train_paths[:config.data.max_images]

    num_train = int(len(train_paths) * 0.8)
    val_paths = train_paths[num_train:]
    train_paths = train_paths[:num_train]

    train_dataset = MoCoDataset(train_paths, train_transform)

    val_dataset = MoCoDataset(val_paths, val_transform)

    logging.info('Initialised ImageNet dataset: Train=%s, Val=%s', len(train_dataset), len(val_dataset))

    return train_dataset, val_dataset
