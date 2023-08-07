import copy

from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad
from torch import nn

from moco.cfg import Config
from moco.model.backbone import get_backbone


class MoCo(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.backbone = get_backbone(config)

        self.projection_head = MoCoProjectionHead(
            list(self.backbone.parameters())[-1].shape[0],
            config.model.hidden_dim,
            config.model.proj_dim
        )

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key
