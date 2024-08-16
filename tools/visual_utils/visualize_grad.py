import cv2
import torch
import numpy as np

fmap_block = list()
input_block = list()


def forward_hook(module, data_input, data_out_put):
    fmap_block.append(data_out_put)
    input_block.append(data_input)


# 该文件主要的目的就是得到最后的中心点关于输入点云的梯度， 生成相应的颜色，用于可视化展示
def init_hook(model, batch_dict):
    batch_dict['voxels'].requires_grad = True
    last_layer = model.dense_head.heads_list[0].hm
    last_layer.register_forward_hook(forward_hook)
    return fmap_block


# 点的坐标格式[batch, x, y, z]，体素坐标的格式[batch, z, y, x]
def get_grad(fmap, batch_dict):
    # 得到其中一类的特征图
    feature_map = fmap[0].features[:, 0].clone()
    feature_map = feature_map.sigmoid()
    topk_score, topk_ind = torch.topk(feature_map, 20)
    center = feature_map[feature_map.shape[0] // 2].sum()
    center.backward(retain_graph=True)
    # 得到体素特征的对应梯度
    grad = torch.abs(batch_dict['voxels'].grad)
    # 对一个体素中的梯度求范数, 也就是某个维度的绝对值之和
    grad = grad.norm(dim=1)
    # 通道之间求范数
    grad = grad.norm(dim=-1).clone()
    grad = (grad - grad.min()) / (grad.max() - grad.min())
    grad = grad.detach().cpu().numpy()
    # 将梯度的值裁剪到 0 ~ 1之间
    grad = np.clip(grad * 10, 0, 1)
    return get_color(batch_dict, grad)


def get_color(batch_dict, grad):
    # 原始点的坐标
    points = batch_dict["points"]
    # 体素的坐标
    coords = batch_dict['voxel_coords']
    point_cloud_range = torch.tensor([0, -40, -3, 70.4, 40, 1]).cuda()
    voxel_size = torch.tensor([0.05, 0.05, 0.1]).cuda()
    grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size
    scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
    scale_yz = grid_size[1] * grid_size[2]
    scale_z = grid_size[2]

    # 点的坐标转换为体素坐标，并且过滤体素范围
    point_coords = torch.floor((points[:, 1:4] - point_cloud_range[0:3]) / voxel_size).int()
    # 给定默认颜色为白色
    colors = np.ones((point_coords.shape[0], 3))
    mask = ((point_coords >= 0) & (point_coords < grid_size)).all(dim=1)
    point_coords = point_coords[mask]
    # 展平索引, 得到每个点的体素一维索引 -----> 得到每个点对应的体素坐标， 后面会根据体素坐标确定每个点的颜色
    index_color = (point_coords[:, 0] * scale_yz + point_coords[:, 1] * scale_z + point_coords[:, 2]).long()

    # 将网格的3维坐标展平为1维的 由于我们只有一帧所以batch维度就不管了， 这里相当于映射规则吧
    merge_coords = coords[:, 1] + coords[:, 2] * scale_z + coords[:, 3] * scale_yz
    heatmap = torch.zeros([scale_xyz.int(), ]).cuda()
    heatmap[merge_coords.long()] = torch.from_numpy(grad).cuda()
    point_colors = heatmap[index_color]

    cam = point_colors
    cam = (cam * 255).to(torch.uint8).cpu().numpy()
    # 映射颜色
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    mask_colors = cam.reshape(-1, 3)
    colors[mask.cpu().numpy().reshape((-1,))] = mask_colors
    return colors

