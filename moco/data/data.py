from torch.utils.data import DataLoader

from moco.cfg import Config
from moco.data.augmentations import get_transform
from moco.data.dummy import get_dummy
from moco.data.el import get_el
from moco.data.imagenet import get_imagenet


def get_loaders(config: Config):
    dataset = config.data.dataset

    train_transform = get_transform(config.data.train_aug, config)
    val_transform = get_transform(config.data.val_aug, config)

    if dataset == 'imagenet':
        train_dataset, val_dataset = get_imagenet(config, train_transform, val_transform)
    elif dataset == 'el':
        train_dataset, val_dataset = get_el(config, train_transform, val_transform)
    elif dataset == 'dummy':
        train_dataset, val_dataset = get_dummy(config, train_transform, val_transform)
    else:
        raise NotImplementedError(f'Dataset {dataset} not supported')

    train_loader = DataLoader(
        train_dataset, batch_size=config.optim.batch_size, drop_last=True,
        shuffle=True, num_workers=config.optim.workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config.optim.batch_size, drop_last=True,
        shuffle=False, num_workers=config.optim.workers, pin_memory=True
    )

    return train_loader, val_loader
