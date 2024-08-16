import torch
import torch_scatter
from torch import nn
from .dynamic_pillar_vfe import PFNLayerV2

class PillarPFNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, use_norm=True, last_layer=False):
        super().__init__()
        self.last_layer = last_layer
        self.use_norm = use_norm
        if not self.last_layer:
            out_channel = out_channel // 2
            self.linear = nn.Linear(in_channel, out_channel)
        else:
            in_channel = in_channel // 2
            self.linear = nn.Sequential(
                nn.Linear(in_channel, out_channel // 2),
                nn.ReLU(),
                nn.Linear(out_channel // 2, out_channel)
            )

        if self.use_norm:
            self.norm = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):
        v = self.linear(inputs)
        x = self.relu(self.norm(v))
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
        if self.last_layer:
            return x_max
        else:
            x = (x + x_max[unq_inv]) / 2
            return x


class PFNLayerBuilder(nn.Module):
    def __init__(self, in_channel, out_channel, use_norm=True, last_layer=False, is_pillar=True):
        super().__init__()
        self.layer = None
        if is_pillar:
            self.layer = PillarPFNLayer(in_channel, out_channel, use_norm, last_layer)
        else:
            self.layer = PFNLayerV2(in_channel, out_channel, use_norm, last_layer)

    def forward(self, inputs, unq_inv):
        return self.layer(inputs, unq_inv)