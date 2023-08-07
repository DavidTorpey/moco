from PIL import Image
from torch.utils.data import Dataset

from moco.cfg import Config


class DummyMoCoDataset(Dataset):
    def __init__(self, config: Config, transform):
        self.transform = transform
        self.config = config

    def __len__(self):
        return 100

    def __getitem__(self, item):
        return self.transform(
            Image.new('RGB', (self.config.data.image_size, self.config.data.image_size)).convert('RGB'))


class MoCoDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        return self.transform(Image.open(self.paths[item]).convert('RGB'))
