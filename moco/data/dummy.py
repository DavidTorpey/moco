import logging

from moco.cfg import Config
from moco.data.dataset import DummyMoCoDataset


def get_dummy(config: Config, train_transform, val_transform):
    train_dataset = DummyMoCoDataset(config, train_transform)

    val_dataset = DummyMoCoDataset(config, val_transform)

    logging.info('Initialised Dummy dataset: Train=%s, Val=%s', len(train_dataset), len(val_dataset))

    return train_dataset, val_dataset
