from lightly.transforms import SimCLRTransform, MoCoV2Transform
import torchvision.transforms as T

from moco.cfg import Config
from moco.constants import MEAN, STD

NORMALIZE = T.Normalize(MEAN, STD)


class BasicTransform:
    def __init__(self, config: Config):
        self.t = T.Compose([
            T.Resize(config.data.image_size),
            T.CenterCrop(config.data.image_size),
            T.ToTensor(),
            NORMALIZE
        ])

    def __call__(self, image):
        return self.t(image)


def get_transform(transform_name, config: Config):
    if transform_name == 'mocov2':
        transform = MoCoV2Transform(
            input_size=config.data.image_size,
            normalize={"mean": MEAN, "std": STD}
        )
    elif transform_name == 'basic':
        transform = BasicTransform(config)
    else:
        raise NotImplementedError(f'Transform {transform_name} not supported.')

    return transform
