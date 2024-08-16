from torch import nn
import torch
import torch_scatter


class PFNLayerV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):

        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]

        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated


class ConvVFE(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, grid_size, window_size=[4, 4], pillar_size=[220, 250],
                 out_channels=3):
        super().__init__()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()
        self.grid_size = torch.tensor(grid_size).cuda()
        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]
        self.pillar_size = torch.tensor(pillar_size).cuda()
        self.pillar_xy = pillar_size[0] * pillar_size[1]
        self.pillar_y = pillar_size[1]
        self.window_size = window_size
        self.pos_weight = nn.Sequential(
            nn.Linear(3, 32, bias=False),
            nn.Linear(32, 3, bias=False)
        )
        self.linear = nn.Linear(3, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()

        num_filters = [10, 64, 128]
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # num_filters = [128, 128]
        # pfn_layers1 = []
        # for i in range(len(num_filters) - 1):
        #     in_filters = num_filters[i]
        #     out_filters = num_filters[i + 1]
        #     pfn_layers1.append(
        #         PFNLayerV2(in_filters, out_filters, last_layer=(i >= len(num_filters) - 2))
        #     )
        # self.pfn_layers1 = nn.ModuleList(pfn_layers1)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    # 根据Link提出的方法，对同一个体素内的点进行聚合，并聚合到均值点上
    def get_voxel_feature(self, points_mean, points_xyz, unq_inv):
        pos_weight = self.pos_weight(points_xyz)
        pos_weight_sin = torch.sin(pos_weight)
        pos_weight_cos = torch.cos(pos_weight)
        # 预处理
        feat_weight_sin = points_xyz * pos_weight_sin
        feat_weight_cos = points_xyz * pos_weight_cos
        # 得到特征之间的和
        feat_add_sin = torch_scatter.scatter_add(feat_weight_sin, unq_inv, dim=0)
        feat_add_cos = torch_scatter.scatter_add(feat_weight_cos, unq_inv, dim=0)

        mean_weight = self.pos_weight(points_mean)
        mean_weight_sin = torch.sin(mean_weight)
        mean_weight_cos = torch.cos(mean_weight)
        center_weight_sin = points_mean * mean_weight_sin
        center_weight_cos = points_mean * mean_weight_cos
        feat_add_sin = feat_add_sin + center_weight_sin
        feat_add_cos = feat_add_cos + center_weight_cos
        final_voxel_feat = feat_add_sin * center_weight_sin + feat_add_cos * center_weight_cos
        return final_voxel_feat

    def forward(self, points):
        return self.forward2(points)

    def forward1(self, points):
        # 1.将点云划分为一个个较小的体素
        points_coords = torch.floor((points[:, [1, 2, 3]] - self.point_cloud_range[[0, 1, 2]]) / self.voxel_size)
        # 2.移除超出范围的点和体素
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1, 2]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()
        # 3.将点云的网格点坐标转换为一维度的
        merge_coords = points[:, 0].int() * self.scale_xyz + points_coords[:, 0] * self.scale_yz + \
                       points_coords[:, 1] * self.scale_z + points_coords[:, 2]
        # 4.根据体素的划分方式，得到每个体素的均值
        unq_coords, unq_inv = torch.unique(merge_coords, return_inverse=True)
        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        # 5.利用Link论文的思想，将一个体素局部的点聚合到均值点中
        voxel_feat = self.get_voxel_feature(points_mean, points_xyz, unq_inv)
        # 最后再次利用上述的方法聚合体素的特征到柱体，unq_coords已经是体素的坐标了， 所以pillar的size肯定是相对于体素的坐标来划分的
        batch = unq_coords // self.scale_xyz
        points_x = (unq_coords % self.scale_xyz) // self.scale_yz
        points_y = (unq_coords % self.scale_yz) // self.scale_z
        pillar_x = points_x // self.window_size[0]
        pillar_y = points_y // self.window_size[1]
        # points_z = unq_coords % self.scale_z
        merge_coords = batch * self.pillar_xy + pillar_x * self.pillar_y + pillar_y
        unq_coords, unq_inv = torch.unique(merge_coords, return_inverse=True)
        pillar_mean = torch_scatter.scatter_mean(voxel_feat, unq_inv, dim=0)
        pillar_feat = self.get_voxel_feature(pillar_mean, voxel_feat, unq_inv)
        unq_coords = unq_coords.int()
        pillar_coords = torch.stack(
            [unq_coords // self.pillar_xy, (unq_coords % self.pillar_xy) // self.pillar_y, unq_coords % self.pillar_y,
             torch.zeros(unq_coords.shape[0]).cuda().int()], dim=-1)
        pillar_coords = pillar_coords[:, [0, 3, 2, 1]]
        pillar_feat = self.linear(pillar_feat)
        pillar_feat = self.norm(pillar_feat)
        pillar_feat = self.relu(pillar_feat)
        return pillar_feat, pillar_coords

    def forward2(self, points):
        # 1.将点云划分为一个个较小的体素
        points_coords = torch.floor((points[:, [1, 2, 3]] - self.point_cloud_range[[0, 1, 2]]) / self.voxel_size).int()
        # 2.移除超出范围的点和体素
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1, 2]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()
        # 3.将点云的网格点坐标转换为一维度的
        merge_coords = points[:, 0].int() * self.scale_xyz + points_coords[:, 0] * self.scale_yz + \
                       points_coords[:, 1] * self.scale_z + points_coords[:, 2]
        # 4.根据体素的划分方式，得到每个体素的均值
        unq_coords, unq_inv = torch.unique(merge_coords, return_inverse=True)
        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)

        # 将点的坐标转换为到每个体素的均值的相对关系
        f_cluster = points_xyz - points_mean[unq_inv, :]
        # 得到每个点体素的中心位置
        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset


        features = [points[:, 1:], f_cluster, f_center]

        # if self.with_distance:
        #     points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
        #     features.append(points_dist)
        features_1 = torch.cat(features, dim=-1)

        # 5.利用Link论文的思想，将一个体素局部的点聚合到均值点中,测试后发现是副作用
        # features_2 = self.get_voxel_feature(points_mean, points_xyz, unq_inv)
        # features = torch.cat([features_1, features_2[unq_inv]], dim=-1)
        features = features_1

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        pillar_feat = features
        unq_coords = unq_coords.int()
        pillar_coords = torch.stack(
            [unq_coords // self.pillar_xy, (unq_coords % self.pillar_xy) // self.pillar_y, unq_coords % self.pillar_y,
             torch.zeros(unq_coords.shape[0]).cuda().int()], dim=-1)
        pillar_coords = pillar_coords[:, [0, 3, 2, 1]]
        return pillar_feat, pillar_coords
