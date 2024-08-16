

# def group_features(inv, features, n):
#     # 最多三个下采样
#     ret_features = features.new_zeros((n, 3, features.shape[1]))
#     key_padding_mask = features.new_zeros((n, 3))
#     for i in range(n):
#         tmp_feature = features.new_zeros((3, features.shape[1]))
#         mask = inv == i
#         cur_feature = features[mask]
#         m = cur_feature.shape[0]
#         tmp_feature[:m] = cur_feature
#         key_padding_mask[i, m:] = 1
#         ret_features[i] = tmp_feature
#     # 转换为bool值, true表示是mask
#     key_padding_mask = key_padding_mask.bool()
#     return tmp_feature, key_padding_mask
#
#
# if __name__ == '__main__':
#     import torch as t
#     n = 3
#     features = t.randn((8,4))
#     order = t.randperm(8)
#     coors = t.tensor([0,0,1,1,1,2,2,2])[order]
#     group_features(coors, features, n)
#     print(features)
#     print(coors)
