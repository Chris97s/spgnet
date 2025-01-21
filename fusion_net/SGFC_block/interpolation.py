import torch

import torch.nn as nn

'''
The purpose of the _get_interpolate function is to extract the feature values at the positions specified in potential_anchor from the feature map. This operation involves two steps:

Determine the coordinates and values of the four neighboring points: First, based on the coordinates in potential_anchor, the function determines the coordinates of four neighboring points: the top-left (lt), top-right (rt), bottom-left (lb), and bottom-right (rb) corners. Then, the feature values corresponding to these four points are extracted from the feature map: vals_lt, vals_rt, vals_lb, and vals_rb.

Bilinear interpolation: Next, bilinear interpolation is used to compute the final interpolated feature value, mapped_vals, based on the coordinates in potential_anchor and the feature values at the four neighboring points. Bilinear interpolation is a commonly used method that performs weighted averaging between the feature values at the four neighboring points, providing an estimate of the feature value at the non-integer coordinate in potential_anchor.

The goal of interpolation is to obtain valid feature values at the non-integer coordinate specified by potential_anchor, providing a more accurate representation of the continuity in the target or image. This is crucial for tasks that require high precision in position, such as image registration or object tracking. Interpolation fills in the gaps between discrete pixels in the feature map, providing smoother feature representations and improving model performance.
'''
class interpolation_layer(nn.Module):
    def __init__(self):
        super(interpolation_layer, self).__init__()

    def forward(self, feature_maps, init_potential_anchor):
        """
        :param feature_map: (Bs, 256, Height, Width)
        :param potential_anchor: (BS, number_point, 2)
        :return:
        """

        feature_dim = feature_maps.size()
        
        # potential_anchor = init_potential_anchor * (feature_dim[2] - 1)
        potential_anchor = init_potential_anchor * (feature_dim[2])

        potential_anchor = torch.clamp(potential_anchor, 0, feature_dim[2] - 1)#截断操作

        anchor_pixel = self._get_interploate(potential_anchor, feature_maps, feature_dim)
        return anchor_pixel


    def _flatten_tensor(self, input):
        return input.contiguous().view(input.nelement())


    def _get_index_point(self, input, anchor, feature_dim):
        index = anchor[:, :, 1] * feature_dim[2] + anchor[:, :, 0]

        output_list = []
        for i in range(feature_dim[0]):
            output_list.append(torch.index_select(input[i].contiguous().flatten(1), 1, index[i]))
        output = torch.stack(output_list)

        return output.permute(0, 2, 1).contiguous()


    def _get_interploate(self, potential_anchor, feature_maps, feature_dim):
        
        #根据 potential_anchor 中的坐标(可能是非整数)，确定了四个相邻的点的坐标，分别是左上角（lt）、右上角（rt）、左下角（lb）、右下角（rb）的坐标
        anchors_lt = potential_anchor.floor().long()
        anchors_rb = potential_anchor.ceil().long()

        anchors_lb = torch.stack([anchors_lt[:, :, 0], anchors_rb[:, :, 1]], 2)
        anchors_rt = torch.stack([anchors_rb[:, :, 0], anchors_lt[:, :, 1]], 2)

        
        #从特征图中获取这四个点对应的特征值，分别是 vals_lt、vals_rt、vals_lb 和 vals_rb。
        vals_lt = self._get_index_point(feature_maps, anchors_lt.detach(), feature_dim)
        vals_rb = self._get_index_point(feature_maps, anchors_rb.detach(), feature_dim)
        vals_lb = self._get_index_point(feature_maps, anchors_lb.detach(), feature_dim)
        vals_rt = self._get_index_point(feature_maps, anchors_rt.detach(), feature_dim)
        
        
        ##双线性插值
        coords_offset_lt = potential_anchor - anchors_lt.type(potential_anchor.data.type())

        vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, :, 0:1]
        vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, :, 0:1]
        mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, :, 1:2]

        return mapped_vals





