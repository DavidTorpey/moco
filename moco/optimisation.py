import logging

from torch.optim import Adam

from moco.cfg import Config


def get_optimiser(parameters, config: Config):
    optimiser_name = config.optim.optimiser

    logging.info('Initialising optimiser: %s', optimiser_name)

    if optimiser_name == 'adam':
        optimiser = Adam(
            parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay
        )
    else:
        raise NotImplementedError(f'Optimiser {optimiser_name} not supported.')

    return optimiser
