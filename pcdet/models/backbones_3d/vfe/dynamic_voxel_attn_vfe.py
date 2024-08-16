import torch
from math import sqrt
from torch import nn
from .vfe_template import VFETemplate

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate

# 为了体素内点的结构信息，使用一种稀疏的注意力方法，来从点提取体素特征
class VoxelAttnVFE(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.linear = nn.Linear(in_channel, out_channel, bias=False)
        self.attn = nn.Sequential(
            nn.Linear(out_channel, out_channel * 4),
            nn.ReLU(),
            nn.Linear(out_channel * 4, out_channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, points, _inv):
        points = self.linear(points)
        attn = self.sigmoid(self.attn(points))
        out = points * attn
        out = torch_scatter.scatter_sum(out, _inv, dim=0)
        return out

class DynamicAttnVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]
        self.out_channel = model_cfg.get("OUT_FILTERS", 16)
        self.voxel_attn = VoxelAttnVFE(3, 4)

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        with torch.no_grad():
            batch_size = batch_dict['batch_size']
            points = batch_dict['points']  # (batch_idx, x, y, z, i, e)

            # # debug
            point_coords = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).long()
            mask = ((point_coords >= 0) & (point_coords < self.grid_size)).all(dim=1)
            points = points[mask]
            point_coords = point_coords[mask]

            merge_coords = points[:, 0].long() * self.scale_xyz + \
                           point_coords[:, 0] * self.scale_yz + \
                           point_coords[:, 1] * self.scale_z + \
                           point_coords[:, 2]
            points_data = points[:, 1:].contiguous()

            unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)

            points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)
            # points_max = torch_scatter.scatter_max(points_data, unq_inv, dim=0)[0]
        # 加入一个注意力机制,学习局部信息
        out = self.voxel_attn(points_data[:, :3] - points_mean[unq_inv, :3], unq_inv)
        out = (out + points_mean) / 2

        unq_coords = unq_coords.long()
        voxel_coords = torch.stack((torch.div(unq_coords, self.scale_xyz, rounding_mode='floor'),
                                    torch.div(unq_coords % self.scale_xyz, self.scale_yz, rounding_mode='floor'),
                                    torch.div(unq_coords % self.scale_yz, self.scale_z, rounding_mode='floor'),
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]].int()

        batch_dict['voxel_features'] = out.contiguous()
        batch_dict['voxel_coords'] = voxel_coords.contiguous()
        return batch_dict
