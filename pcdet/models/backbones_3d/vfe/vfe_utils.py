import torch_scatter
from torch import nn
import torch
# from .dynamic_pillar_vfe import PFNLayerV2

class PFNLayerV9(nn.Module):
    def __init__(self, in_channel, out_channel, use_norm=True, last_layer=False):
        super().__init__()
        self.last_layer = last_layer
        self.use_norm = use_norm
        if not self.last_layer:
            out_channel = out_channel // 2
        # self.q = nn.Linear(out_channel, out_channel)
        self.kv = nn.Linear(in_channel, out_channel)
        self.w = nn.Sequential(nn.Linear(in_channel, out_channel),
                               nn.BatchNorm1d(out_channel),
                               nn.ReLU(),
                               nn.Linear(out_channel, out_channel))
        if self.use_norm:
            self.norm = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):
        v = self.kv(inputs)

        x = self.relu(self.norm(v))
        if self.last_layer:
            return x
        else:
            # 得到最大的数据后，使用注意力的方式聚集其他点的信息
            q_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
            # 得到基于注意力的加权求和
            weight = self.w(inputs)
            soft_weight = torch_scatter.scatter_softmax(weight, unq_inv, dim=0)
            weight_x = torch_scatter.scatter_mean(soft_weight * v, unq_inv, dim=0)
            res = torch.cat([weight_x, q_max], dim=-1)
            return res


class LinkConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pre_mix = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        )
        self.activate = nn.ReLU(True)
        self.pos_weight = nn.Linear(out_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, feat, points_xyz, unq_inv):
        points_xyz = self.pre_mix(points_xyz)
        pos_weight = self.pos_weight(points_xyz)
        pos_weight_sin = torch.sin(pos_weight)
        pos_weight_cos = torch.cos(pos_weight)
        # 预处理
        feat_weight_sin = points_xyz * pos_weight_sin
        feat_weight_cos = points_xyz * pos_weight_cos
        # 得到特征之间的和
        feat_add_sin = torch_scatter.scatter_add(feat_weight_sin, unq_inv, dim=0)
        feat_add_cos = torch_scatter.scatter_add(feat_weight_cos, unq_inv, dim=0)

        feat = self.pre_mix(feat)
        mean_weight = self.pos_weight(feat)
        mean_weight_sin = torch.sin(mean_weight)
        mean_weight_cos = torch.cos(mean_weight)
        center_weight_sin = feat * mean_weight_sin
        center_weight_cos = feat * mean_weight_cos
        feat_add_sin = feat_add_sin + center_weight_sin
        feat_add_cos = feat_add_cos + center_weight_cos
        final_voxel_feat = feat_add_sin * center_weight_sin + feat_add_cos * center_weight_cos
        final_voxel_feat = self.activate(self.norm(final_voxel_feat))
        return final_voxel_feat


class PFNLayerV10(nn.Module):
    def __init__(self, in_channel, out_channel, use_norm=True, last_layer=False):
        super().__init__()
        self.last_layer = last_layer
        self.use_norm = use_norm
        if not self.last_layer:
            out_channel = out_channel // 2
        # self.q = nn.Linear(out_channel, out_channel)
        self.kv = nn.Linear(in_channel, out_channel)
        self.w = nn.Sequential(nn.Linear(in_channel, out_channel),
                               nn.BatchNorm1d(out_channel),
                               nn.ReLU(),
                               nn.Linear(out_channel, out_channel))
        if self.use_norm:
            self.norm = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, inputs, unq_inv):
        v = self.kv(inputs)

        x = self.relu(self.norm(v))
        if self.last_layer:
            return x
        else:
            # 得到最大的数据后，使用注意力的方式聚集其他点的信息
            q_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
            # 得到基于注意力的加权求和
            weight = self.w(inputs)
            # 得到每个点的注意力权重
            soft_weight = torch_scatter.scatter_softmax(weight, unq_inv, dim=0)
            weight_x = torch_scatter.scatter_mean(soft_weight * v, unq_inv, dim=0)
            weight_max = torch_scatter.scatter_max(soft_weight * v, unq_inv, dim=0)[0]
            q_max = (q_max + weight_max) / 2
            res = torch.cat([weight_x, q_max], dim=-1)
            return res


class PFNLayerV11(nn.Module):
    def __init__(self, in_channel, out_channel, use_norm=True, last_layer=False):
        super().__init__()
        self.last_layer = last_layer
        self.use_norm = use_norm
        if not self.last_layer:
            out_channel = out_channel // 2
        # self.q = nn.Linear(out_channel, out_channel)
        self.kv = nn.Linear(in_channel, out_channel)
        self.w = nn.Sequential(nn.Linear(in_channel, out_channel),
                               nn.BatchNorm1d(out_channel),
                               nn.ReLU(),
                               nn.Linear(out_channel, out_channel))
        if self.use_norm:
            self.norm = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.link_conv = LinkConv(in_channel, out_channel)

    def forward(self, inputs, unq_inv):
        v = self.kv(inputs)

        x = self.relu(self.norm(v))
        if self.last_layer:
            return x
        else:
            # 得到最大的数据后，使用注意力的方式聚集其他点的信息
            q_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
            # 得到基于注意力的加权求和
            weight = self.w(inputs)
            # 得到每个点的注意力权重
            soft_weight = torch_scatter.scatter_softmax(weight, unq_inv, dim=0)
            weight_x = torch_scatter.scatter_mean(soft_weight * v, unq_inv, dim=0)
            link_feat = self.link_conv(q_max, inputs, unq_inv)
            weight_x = (link_feat + weight_x) / 2
            res = torch.cat([weight_x, q_max], dim=-1)
            return res


class PFNLayerV12(nn.Module):
    def __init__(self, in_channel, out_channel, use_norm=True, last_layer=False):
        super().__init__()
        self.last_layer = last_layer
        self.use_norm = use_norm
        if not self.last_layer:
            out_channel = out_channel // 2
        # self.q = nn.Linear(out_channel, out_channel)
        self.kv = nn.Sequential(nn.Linear(in_channel, out_channel),
                                # nn.BatchNorm1d(out_channel),
                                # nn.ReLU(),
                                # nn.Linear(out_channel, out_channel)
                                )
        self.w = nn.Sequential(nn.Linear(in_channel, out_channel),
                               nn.BatchNorm1d(out_channel),
                               nn.ReLU(),
                               nn.Linear(out_channel, out_channel))
        if self.use_norm:
            self.norm = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.link_conv = LinkConv(in_channel, out_channel)

    def forward(self, inputs, unq_inv):
        v = self.kv(inputs)

        x = self.relu(self.norm(v))
        if self.last_layer:
            return x
        else:
            # 得到最大的数据后，使用注意力的方式聚集其他点的信息
            q_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
            # 得到基于注意力的加权求和
            weight = self.w(inputs)
            # 得到每个点的注意力权重
            soft_weight = torch_scatter.scatter_softmax(weight, unq_inv, dim=0)
            weight_x = torch_scatter.scatter_mean(soft_weight * v, unq_inv, dim=0)
            link_feat = self.link_conv(weight_x, inputs, unq_inv)
            weight_x = (link_feat + weight_x) / 2
            res = torch.cat([weight_x, q_max], dim=-1)
            return res


class PFNLayerV13(nn.Module):
    def __init__(self, in_channel, out_channel, use_norm=True, last_layer=False):
        super().__init__()
        self.last_layer = last_layer
        self.use_norm = use_norm
        if not self.last_layer:
            out_channel = out_channel // 2
        self.kv = nn.Linear(in_channel, out_channel)
        self.w = nn.Sequential(nn.Linear(in_channel, out_channel),
                               nn.BatchNorm1d(out_channel),
                               nn.ReLU(),
                               nn.Linear(out_channel, out_channel))
        if self.use_norm:
            self.norm = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.link_conv = LinkConv(in_channel, out_channel)

    def forward(self, inputs, unq_inv):
        v = self.kv(inputs)

        x = self.relu(self.norm(v))
        if self.last_layer:
            x = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
            return x
        else:
            # 得到最大的数据后，使用注意力的方式聚集其他点的信息
            q_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
            # 得到基于注意力的加权求和
            weight = self.w(inputs)
            # 得到每个点的注意力权重
            soft_weight = torch_scatter.scatter_softmax(weight, unq_inv, dim=0)
            weight_x = torch_scatter.scatter_mean(soft_weight * v, unq_inv, dim=0)
            link_feat = self.link_conv(q_max, inputs, unq_inv)
            weight_x = (link_feat + weight_x + q_max) / 3
            res = torch.cat([x, weight_x[unq_inv]], dim=-1)
            return res


class PFNLayerV14(nn.Module):
    def __init__(self, in_channel, out_channel, use_norm=True, last_layer=False):
        super().__init__()
        self.last_layer = last_layer
        self.use_norm = use_norm
        if not self.last_layer:
            out_channel = out_channel // 2
        # self.q = nn.Linear(out_channel, out_channel)
        self.kv = nn.Linear(in_channel, out_channel)
        self.w = nn.Sequential(nn.Linear(in_channel, out_channel),
                               nn.BatchNorm1d(out_channel),
                               nn.ReLU(),
                               nn.Linear(out_channel, 1))
        if self.use_norm:
            self.norm = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):
        v = self.kv(inputs)

        x = self.relu(self.norm(v))
        if self.last_layer:
            return x
        else:
            # 得到最大的数据后，使用注意力的方式聚集其他点的信息
            q_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
            # 得到基于注意力的加权求和, 我认为权重的混合应该在每个点上进行，而不是每个点的每个通道上进行，所以进行相加
            weight = self.w(inputs)
            soft_weight = torch_scatter.scatter_softmax(weight, unq_inv, dim=0)
            weight_x = torch_scatter.scatter_mean(soft_weight * v, unq_inv, dim=0)
            res = torch.cat([weight_x, q_max], dim=-1)
            return res


class ChannelAttn(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.fc1 = nn.Linear(in_channel, in_channel * 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channel * 2, out_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, points, unq_inv):
        min_feat = torch_scatter.scatter_min(points, unq_inv, dim=0)[0]
        weight = self.fc2(self.relu(self.fc1(min_feat)))
        return self.sigmoid(weight)


class ChannelAttn2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.fc1 = nn.Linear(in_channel, in_channel * 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channel * 2, out_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, points, unq_inv):
        max_feat = torch_scatter.scatter_max(points, unq_inv, dim=0)[0]
        mean_feat = torch_scatter.scatter_mean(points, unq_inv, dim=0)
        max_out = self.fc2(self.relu(self.fc1(max_feat)))
        mean_out = self.fc2(self.relu(self.fc1(mean_feat)))
        return self.sigmoid(max_out + mean_out)


class PFNLayerV15(nn.Module):
    def __init__(self, in_channel, out_channel, use_norm=True, last_layer=False):
        super().__init__()
        self.last_layer = last_layer
        self.use_norm = use_norm
        if not self.last_layer:
            out_channel = out_channel // 2
        self.linear = nn.Linear(in_channel, out_channel)
        self.w = nn.Sequential(nn.Linear(in_channel, out_channel),
                               nn.BatchNorm1d(out_channel),
                               nn.ReLU(),
                               nn.Linear(out_channel, 1))
        if self.use_norm:
            self.norm = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.channel_attn = ChannelAttn(out_channel, out_channel)

    def forward(self, inputs, unq_inv):
        v = self.linear(inputs)
        norm_v = self.norm(v)
        x = self.relu(norm_v)
        if self.last_layer:
            return torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
        else:
            # 得到最大的数据后，使用注意力的方式聚集其他点的信息
            q_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
            # 得到基于注意力的加权求和, 我认为权重的混合应该在每个点上进行，而不是每个点的每个通道上进行，参数更少，好优化
            weight = self.w(inputs)
            soft_weight = torch_scatter.scatter_softmax(weight, unq_inv, dim=0)
            weight_x = torch_scatter.scatter_mean(soft_weight * v, unq_inv, dim=0)
            channel_weight = self.channel_attn(norm_v, unq_inv)
            final_feat = channel_weight * q_max + (1 - channel_weight) * weight_x
            return torch.cat([x, final_feat[unq_inv]], dim=-1)


# 如何保留一部分高度相关的特征
class PFNLayerV16(nn.Module):
    def __init__(self, in_channel, out_channel, use_norm=True, last_layer=False):
        super().__init__()
        self.last_layer = last_layer
        self.use_norm = use_norm
        if not self.last_layer:
            out_channel = out_channel // 2
        self.linear = nn.Linear(in_channel, out_channel)
        self.w = nn.Sequential(nn.Linear(in_channel, out_channel),
                               nn.BatchNorm1d(out_channel),
                               nn.ReLU(),
                               nn.Linear(out_channel, 1))
        if self.use_norm:
            self.norm = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.channel_attn = ChannelAttn(out_channel, out_channel)

        # 尝试加强行人的识别能力
        # self.channel_attn2 = ChannelAttn(out_channel, out_channel)
        self.link = LinkConv(out_channel, out_channel)

    def forward(self, inputs, unq_inv):
        v = self.linear(inputs)
        norm_v = self.norm(v)
        x = self.relu(norm_v)
        if self.last_layer:
            return torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
        else:
            # 得到最大的数据后，使用注意力的方式聚集其他点的信息
            q_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
            # 得到基于注意力的加权求和, 我认为权重的混合应该在每个点上进行，而不是每个点的每个通道上进行，参数更少，好优化
            weight = self.w(inputs)
            soft_weight = torch_scatter.scatter_softmax(weight, unq_inv, dim=0)
            weight_x = torch_scatter.scatter_mean(soft_weight * v, unq_inv, dim=0)
            channel_weight = self.channel_attn(norm_v, unq_inv)
            final_feat = channel_weight * q_max + (1 - channel_weight) * weight_x
            out = self.link(weight_x, x, unq_inv)
            x = (x + out[unq_inv]) / 2
            return torch.cat([x, final_feat[unq_inv]], dim=-1)


class PFNLayerV17(nn.Module):
    def __init__(self, in_channel, out_channel, use_norm=True, last_layer=False):
        super().__init__()
        self.last_layer = last_layer
        self.use_norm = use_norm
        if not self.last_layer:
            out_channel = out_channel // 2
        self.linear = nn.Linear(in_channel, out_channel)
        self.w = nn.Sequential(nn.Linear(in_channel, out_channel),
                               nn.BatchNorm1d(out_channel),
                               nn.ReLU(),
                               nn.Linear(out_channel, 1))
        if self.use_norm:
            self.norm = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.channel_attn = ChannelAttn(out_channel, out_channel)

        # 尝试加强行人的识别能力
        # self.channel_attn2 = ChannelAttn(out_channel, out_channel)
        self.link1 = LinkConv(out_channel, out_channel)
        self.link2 = LinkConv(out_channel, out_channel)

    def forward(self, inputs, unq_inv):
        v = self.linear(inputs)
        norm_v = self.norm(v)
        x = self.relu(norm_v)
        if self.last_layer:
            # self.link(, x, unq_inv)
            # mean_feat = torch_scatter.scatter_mean(x, unq_inv, dim=0)
            # max_feat = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
            # return (mean_feat + max_feat) / 2
            return torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
        else:
            # 得到最大的数据后，使用注意力的方式聚集其他点的信息
            q_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
            # 得到基于注意力的加权求和, 我认为权重的混合应该在每个点上进行，而不是每个点的每个通道上进行，参数更少，好优化
            weight = self.w(inputs)
            soft_weight = torch_scatter.scatter_softmax(weight, unq_inv, dim=0)
            weight_x = torch_scatter.scatter_mean(soft_weight * v, unq_inv, dim=0)
            channel_weight = self.channel_attn(norm_v, unq_inv)
            final_feat = channel_weight * q_max + (1 - channel_weight) * weight_x
            out = self.link1(weight_x, x, unq_inv)
            out = self.link2(out, x, unq_inv)
            x = (x + out[unq_inv]) / 2
            return torch.cat([x, final_feat[unq_inv]], dim=-1)


class PFNLayerV18(nn.Module):
    def __init__(self, in_channel, out_channel, use_norm=True, last_layer=False):
        super().__init__()
        self.last_layer = last_layer
        self.use_norm = use_norm
        if not self.last_layer:
            out_channel = out_channel // 2
        else:
            self.w = nn.Sequential(nn.Linear(out_channel, out_channel),
                                   nn.BatchNorm1d(out_channel),
                                   nn.ReLU(),
                                   nn.Linear(out_channel, 1))
            self.channel_attn = ChannelAttn(out_channel, out_channel)

            # 尝试加强行人的识别能力
            self.link1 = LinkConv(out_channel, out_channel)

        if self.use_norm:
            self.norm = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_channel, out_channel)

    def forward(self, inputs, unq_inv):
        v = self.linear(inputs)
        x = self.relu(self.norm(v))
        if self.last_layer:
            weight = self.w(x)
            soft_weight = torch_scatter.scatter_softmax(weight, unq_inv, dim=0)
            weight_x = soft_weight * x
            mean_feat = torch_scatter.scatter_mean(weight_x, unq_inv, dim=0)
            link_feat = self.link1(mean_feat, x, unq_inv)
            mean_feat = (mean_feat + link_feat) / 2
            max_feat = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
            # channel_feat = (mean_feat + max_feat) / 2
            # channel_weight = self.channel_attn(x, unq_inv)
            return (mean_feat + max_feat) / 2
        else:
            max_feat = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
            return torch.cat([x, max_feat[unq_inv]], dim=-1)


# 函数功能， 获取每个柱体的最大和最小的xyz
def getBorder(pointxyz, unq_inv):
    max_xyz = torch_scatter.scatter_max(pointxyz, unq_inv, dim=0)


# 在pillar中实现link类型的大核卷积
class LinkConvInPillar(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pre_mix = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        )

        self.activate = nn.ReLU(True)
        # 根据位置，生成权重
        self.pos_weight1 = nn.Linear(3, out_channels)
        self.pos_weight2 = nn.Linear(3, out_channels)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

    # feat_all [n, c]
    def forward(self, points_xyz, feat_all, unq_inv):
        points_xyz = points_xyz.floor()
        feat_all = self.pre_mix(feat_all)
        pos_weight1 = self.pos_weight1(points_xyz)
        pos_weight2 = self.pos_weight2(points_xyz)

        feat_weight = pos_weight1 * feat_all
        feat_add = torch_scatter.scatter_add(feat_weight, unq_inv, dim=0)
        feat_add = feat_add[unq_inv]

        out = pos_weight2 * feat_all - feat_add
        out = self.activate(self.norm(out))
        return out


class PFNLayerV19(nn.Module):
    def __init__(self, in_channel, out_channel, last_layer=False, first_layer=False):
        super().__init__()
        self.last_layer = last_layer
        self.first_layer = first_layer
        if self.first_layer:
            links = [LinkConvInPillar2(in_channel, out_channel),
                     LinkConvInPillar2(out_channel, out_channel),
                     LinkConvInPillar2(out_channel, out_channel)]
            self.links = nn.ModuleList(links)
            in_channel = out_channel
        if not self.last_layer and not first_layer:
            out_channel = out_channel // 2
        self.norm = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_channel, out_channel)

    def forward(self, inputs, unq_inv):
        if self.first_layer:
            feat = inputs
            points_xyz = inputs[:, 3:6]
            for link in self.links:
                feat = link(points_xyz, feat, unq_inv)
            return self.relu(self.norm(self.linear(feat)))
        feat = self.relu(self.norm(self.linear(inputs)))
        max_feat = torch_scatter.scatter_max(feat, unq_inv, dim=0)[0]
        if self.last_layer:
            return max_feat
        mean_feat = torch_scatter.scatter_mean(feat, unq_inv, dim=0)
        feat2 = (max_feat + mean_feat) / 2
        return torch.cat([feat, feat2[unq_inv]], dim=-1)


class LinkConvInPillar2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pre_mix = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        )
        group = 1
        self.group = group

        self.activate = nn.ReLU(True)
        # 根据位置，生成权重
        # self.pos_weight1 = nn.Linear(3, out_channels // group)
        self.pos_weight2 = nn.Linear(3, out_channels // group)
        self.offset_weight = nn.Linear(out_channels // group, out_channels // group)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

    # feat_all [n, c]
    def forward(self, points_xyz, feat_all, unq_inv):
        points_xyz = points_xyz.floor()
        feat_all = self.pre_mix(feat_all)
        # pos_weight1 = self.pos_weight1(points_xyz)
        pos_weight2 = self.pos_weight2(points_xyz)
        pos_weight2 = pos_weight2.repeat(1, self.group)
        tmp_feat = pos_weight2 * feat_all
        feat_add = torch_scatter.scatter_add(tmp_feat, unq_inv, dim=0)

        tmp_weight = torch_scatter.scatter_add(pos_weight2, unq_inv, dim=0)

        out = pos_weight2 * (-feat_add[unq_inv]) / (tmp_weight[unq_inv] + 1e-4)
        out = self.activate(self.norm(out))
        # 残差

        out = out + feat_all
        return out


class PillarPFNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, use_norm=True, last_layer=False):
        super().__init__()
        self.last_layer = last_layer
        self.use_norm = use_norm
        if not self.last_layer:
            out_channel = out_channel // 2
        self.linear = nn.Linear(in_channel, out_channel)
        self.w = nn.Sequential(nn.Linear(in_channel, out_channel),
                               nn.BatchNorm1d(out_channel),
                               nn.ReLU(),
                               nn.Linear(out_channel, out_channel))
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
            # 得到基于注意力的加权求和
            weight = self.w(inputs)
            soft_weight = torch_scatter.scatter_softmax(weight, unq_inv, dim=0)
            # 保留稍微多一点的细节信息
            weight_x = torch_scatter.scatter_mean(soft_weight * x, unq_inv, dim=0)
            return torch.cat([weight_x, x_max], dim=-1)


