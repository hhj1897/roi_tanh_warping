import math
import torch
import torch.nn.functional as tf
from typing import Tuple, Union


__all__ = ['make_square_rois',
           'roi_tanh_warp', 'roi_tanh_restore',
           'roi_tanh_polar_warp', 'roi_tanh_polar_restore',
           'roi_tanh_circular_warp', 'roi_tanh_circular_restore', 
           'get_warp_func', 'get_restore_func']


def arctanh(x: torch.Tensor) -> torch.Tensor:
    return torch.log((1.0 + x) / (1.0 - x).clamp(1e-9)) / 2.0


def make_square_rois(rois: torch.Tensor, opt=0) -> torch.Tensor:
    roi_sizes = (rois[..., 2:4] - rois[..., :2])
    if opt < 0:
        target_sizes = roi_sizes.min(dim=-1)[0]
    elif opt > 0:
        target_sizes = roi_sizes.max(dim=-1)[0]
    else:
        target_sizes = (roi_sizes[..., 0] * roi_sizes[..., 1]) ** 0.5
    deltas = (roi_sizes - target_sizes) / 2.0
    deltas = torch.cat((deltas, -deltas), -1)
    return rois + deltas


def roi_tanh_warp(images: torch.Tensor, rois: torch.Tensor, target_size: Tuple[int, int],
                  angular_offsets: Union[float, torch.Tensor] = 0.0, interpolation: str = 'bilinear',
                  padding: str = 'zeros') -> torch.Tensor:
    image_height, image_width = images.size()[-2:]
    roi_centers = (rois[:, 2:4] + rois[:, :2]) / 2.0
    half_roi_sizes = (rois[:, 2:4] - rois[:, :2]) / 2.0

    grids = torch.zeros(images.size()[:1] + target_size + (2,), dtype=images.dtype, device=images.device)
    warped_x_indices = arctanh(torch.arange(-1.0, 1.0, 2.0 / target_size[1], dtype=grids.dtype,
                                            device=grids.device) + 1.0 / target_size[1])
    warped_y_indices = arctanh(torch.arange(-1.0, 1.0, 2.0 / target_size[0], dtype=grids.dtype,
                                            device=grids.device) + 1.0 / target_size[0])

    if torch.is_tensor(angular_offsets):
        cos_offsets, sin_offsets = angular_offsets.cos(), angular_offsets.sin()
    else:
        cos_offsets = [math.cos(angular_offsets)] * grids.size()[0]
        sin_offsets = [math.sin(angular_offsets)] * grids.size()[0]

    for roi_center, half_roi_size, grid, cos_offset, sin_offset in zip(roi_centers, half_roi_sizes, grids,
                                                                       cos_offsets, sin_offsets):
        src_x_indices = (half_roi_size[0] * warped_x_indices).unsqueeze(0).expand(target_size)
        src_y_indices = (half_roi_size[1] * warped_y_indices).unsqueeze(-1).expand(target_size)
        src_x_indices, src_y_indices = (cos_offset * src_x_indices - sin_offset * src_y_indices,
                                        cos_offset * src_y_indices + sin_offset * src_x_indices)
        grid[..., 0] = (roi_center[0] + src_x_indices) / (image_width - 1.0) * 2.0 - 1.0
        grid[..., 1] = (roi_center[1] + src_y_indices) / (image_height - 1.0) * 2.0 - 1.0

    return tf.grid_sample(images, grids, mode=interpolation, padding_mode=padding, align_corners=True)


def roi_tanh_restore(warped_images: torch.Tensor, rois: torch.Tensor, image_size: Tuple[int, int],
                     angular_offsets: Union[float, torch.Tensor] = 0.0, interpolation: str = 'bilinear',
                     padding: str = 'zeros') -> torch.Tensor:
    warped_height, warped_width = warped_images.size()[-2:]
    roi_centers = (rois[:, 2:4] + rois[:, :2]) / 2.0
    half_roi_sizes = (rois[:, 2:4] - rois[:, :2]) / 2.0

    grids = torch.zeros(warped_images.size()[:1] + image_size + (2,),
                        dtype=warped_images.dtype, device=warped_images.device)
    dest_x_indices = torch.arange(image_size[1], dtype=grids.dtype, device=grids.device)
    dest_y_indices = torch.arange(image_size[0], dtype=grids.dtype, device=grids.device)

    if torch.is_tensor(angular_offsets):
        cos_offsets, sin_offsets = angular_offsets.cos(), angular_offsets.sin()
    else:
        cos_offsets = [math.cos(angular_offsets)] * grids.size()[0]
        sin_offsets = [math.sin(angular_offsets)] * grids.size()[0]

    for roi_center, half_roi_size, grid, cos_offset, sin_offset in zip(roi_centers, half_roi_sizes, grids,
                                                                       cos_offsets, sin_offsets):
        src_x_indices = (dest_x_indices - roi_center[0]).unsqueeze(0).expand(image_size)
        src_y_indices = (dest_y_indices - roi_center[1]).unsqueeze(-1).expand(image_size)
        src_x_indices, src_y_indices = (cos_offset * src_x_indices + sin_offset * src_y_indices,
                                        cos_offset * src_y_indices - sin_offset * src_x_indices)
        src_x_indices = (torch.tanh(src_x_indices / half_roi_size[0]) + 1.0) / 2.0 * warped_width - 0.5
        src_y_indices = (torch.tanh(src_y_indices / half_roi_size[1]) + 1.0) / 2.0 * warped_height - 0.5
        grid[..., 0] = (src_x_indices / (warped_width - 1.0) * 2.0 - 1.0)
        grid[..., 1] = (src_y_indices / (warped_height - 1.0) * 2.0 - 1.0)

    return tf.grid_sample(warped_images, grids, mode=interpolation, padding_mode=padding, align_corners=True)


def roi_tanh_polar_warp(images: torch.Tensor, rois: torch.Tensor, target_size: Tuple[int, int],
                        angular_offsets: Union[float, torch.Tensor] = 0.0, interpolation: str = 'bilinear',
                        padding: str = 'zeros') -> torch.Tensor:
    image_height, image_width = images.size()[-2:]
    roi_centers = (rois[:, 2:4] + rois[:, :2]) / 2.0
    rois_radii = (rois[:, 2:4] - rois[:, :2]) / math.pi ** 0.5

    grids = torch.zeros(images.size()[:1] + target_size + (2,), dtype=images.dtype, device=images.device)
    wapred_radii = arctanh(torch.arange(0.0, 1.0, 1.0 / target_size[1], dtype=grids.dtype,
                                        device=grids.device)).unsqueeze(0).expand(target_size)
    thetas = torch.arange(0.0, 2.0 * math.pi, 2.0 * math.pi / target_size[0], dtype=grids.dtype, device=grids.device)

    if torch.is_tensor(angular_offsets):
        cos_offsets, sin_offsets = angular_offsets.cos(), angular_offsets.sin()
    else:
        cos_offsets = [math.cos(angular_offsets)] * grids.size()[0]
        sin_offsets = [math.sin(angular_offsets)] * grids.size()[0]

    for roi_center, roi_radii, grid, cos_offset, sin_offset in zip(roi_centers, rois_radii, grids,
                                                                   cos_offsets, sin_offsets):
        warped_x_indices = roi_radii[0] * wapred_radii * torch.cos(thetas).unsqueeze(-1).expand(target_size)
        warped_y_indices = roi_radii[1] * wapred_radii * torch.sin(thetas).unsqueeze(-1).expand(target_size)
        src_x_indices, src_y_indices = (cos_offset * warped_x_indices - sin_offset * warped_y_indices,
                                        cos_offset * warped_y_indices + sin_offset * warped_x_indices)
        grid[..., 0] = (roi_center[0] + src_x_indices) / (image_width - 1.0) * 2.0 - 1.0
        grid[..., 1] = (roi_center[1] + src_y_indices) / (image_height - 1.0) * 2.0 - 1.0

    return tf.grid_sample(images, grids, mode=interpolation, padding_mode=padding, align_corners=True)


def roi_tanh_polar_restore(warped_images: torch.Tensor, rois: torch.Tensor, image_size: Tuple[int, int],
                           angular_offsets: Union[float, torch.Tensor] = 0.0, interpolation: str = 'bilinear',
                           padding: str = 'zeros') -> torch.Tensor:
    warped_height, warped_width = warped_images.size()[-2:]
    roi_centers = (rois[:, 2:4] + rois[:, :2]) / 2.0
    rois_radii = (rois[:, 2:4] - rois[:, :2]) / math.pi ** 0.5

    grids = torch.zeros(warped_images.size()[:1] + image_size + (2,),
                        dtype=warped_images.dtype, device=warped_images.device)
    dest_x_indices = torch.arange(image_size[1], dtype=warped_images.dtype, device=warped_images.device)
    dest_y_indices = torch.arange(image_size[0], dtype=warped_images.dtype, device=warped_images.device)
    dest_indices = torch.cat((dest_x_indices.unsqueeze(0).expand(image_size).unsqueeze(-1),
                              dest_y_indices.unsqueeze(-1).expand(image_size).unsqueeze(-1)), -1)

    if torch.is_tensor(angular_offsets):
        cos_offsets, sin_offsets = angular_offsets.cos(), angular_offsets.sin()
    else:
        cos_offsets = [math.cos(angular_offsets)] * grids.size()[0]
        sin_offsets = [math.sin(angular_offsets)] * grids.size()[0]

    warped_images = tf.pad(tf.pad(warped_images, [0, 0, 1, 1], mode='circular'), [1, 0, 0, 0], mode='replicate')
    for roi_center, roi_radii, grid, cos_offset, sin_offset in zip(roi_centers, rois_radii, grids,
                                                                   cos_offsets, sin_offsets):
        normalised_dest_indices = dest_indices - roi_center
        normalised_dest_indices[..., 0], normalised_dest_indices[..., 1] = (
            cos_offset * normalised_dest_indices[..., 0] + sin_offset * normalised_dest_indices[..., 1],
            cos_offset * normalised_dest_indices[..., 1] - sin_offset * normalised_dest_indices[..., 0])
        normalised_dest_indices /= roi_radii
        radii = normalised_dest_indices.norm(dim=-1)
        grid[..., 0] = (torch.tanh(radii) * 2.0 * warped_width + 2) / warped_width - 1.0
        grid[..., 1] = ((torch.atan2(normalised_dest_indices[..., 1], normalised_dest_indices[..., 0]) /
                         math.pi).remainder(2.0) * warped_height + 2) / (warped_height + 1.0) - 1.0

    return tf.grid_sample(warped_images, grids, mode=interpolation, padding_mode=padding, align_corners=True)


def roi_tanh_circular_warp(images: torch.Tensor, rois: torch.Tensor, target_size: Tuple[int, int],
                           angular_offsets: Union[float, torch.Tensor] = 0.0, interpolation: str = 'bilinear',
                           padding: str = 'zeros') -> torch.Tensor:
    image_height, image_width = images.size()[-2:]
    roi_centers = (rois[:, 2:4] + rois[:, :2]) / 2.0
    rois_radii = (rois[:, 2:4] - rois[:, :2]) / math.pi ** 0.5

    grids = torch.zeros(images.size()[:1] + target_size + (2,), dtype=images.dtype, device=images.device)
    normalised_dest_x_indices = torch.arange(-1.0, 1.0, 2.0 / target_size[1], dtype=grids.dtype,
                                             device=grids.device) + 1.0 / target_size[1]
    normalised_dest_y_indices = torch.arange(-1.0, 1.0, 2.0 / target_size[0], dtype=grids.dtype,
                                             device=grids.device) + 1.0 / target_size[0]
    normalised_dest_indices = torch.cat((normalised_dest_x_indices.unsqueeze(0).expand(target_size).unsqueeze(-1),
                                         normalised_dest_y_indices.unsqueeze(-1).expand(target_size).unsqueeze(-1)),
                                        -1)
    radii = normalised_dest_indices.norm(dim=-1)
    orientation_x = normalised_dest_indices[..., 0] / radii.clamp(min=1e-9)
    orientation_y = normalised_dest_indices[..., 1] / radii.clamp(min=1e-9)

    if torch.is_tensor(angular_offsets):
        cos_offsets, sin_offsets = angular_offsets.cos(), angular_offsets.sin()
    else:
        cos_offsets = [math.cos(angular_offsets)] * grids.size()[0]
        sin_offsets = [math.sin(angular_offsets)] * grids.size()[0]

    warped_radii = arctanh(radii)
    warped_x_indices = warped_radii * orientation_x
    warped_y_indices = warped_radii * orientation_y
    for roi_center, roi_radii, grid, cos_offset, sin_offset in zip(roi_centers, rois_radii, grids,
                                                                   cos_offsets, sin_offsets):
        src_x_indices, src_y_indices = roi_radii[0] * warped_x_indices, roi_radii[1] * warped_y_indices
        src_x_indices, src_y_indices = (cos_offset * src_x_indices - sin_offset * src_y_indices,
                                        cos_offset * src_y_indices + sin_offset * src_x_indices)
        grid[..., 0] = (roi_center[0] + src_x_indices) / (image_width - 1.0) * 2.0 - 1.0
        grid[..., 1] = (roi_center[1] + src_y_indices) / (image_height - 1.0) * 2.0 - 1.0

    return tf.grid_sample(images, grids, mode=interpolation, padding_mode=padding, align_corners=True)


def roi_tanh_circular_restore(warped_images: torch.Tensor, rois: torch.Tensor, image_size: Tuple[int, int],
                              angular_offsets: Union[float, torch.Tensor] = 0.0, interpolation: str = 'bilinear',
                              padding: str = 'zeros') -> torch.Tensor:
    warped_height, warped_width = warped_images.size()[-2:]
    roi_centers = (rois[:, 2:4] + rois[:, :2]) / 2.0
    rois_radii = (rois[:, 2:4] - rois[:, :2]) / math.pi ** 0.5

    grids = torch.zeros(warped_images.size()[:1] + image_size + (2,),
                        dtype=warped_images.dtype, device=warped_images.device)
    dest_x_indices = torch.arange(image_size[1], dtype=warped_images.dtype, device=warped_images.device)
    dest_y_indices = torch.arange(image_size[0], dtype=warped_images.dtype, device=warped_images.device)
    dest_indices = torch.cat((dest_x_indices.unsqueeze(0).expand(image_size).unsqueeze(-1),
                              dest_y_indices.unsqueeze(-1).expand(image_size).unsqueeze(-1)), -1)

    if torch.is_tensor(angular_offsets):
        cos_offsets, sin_offsets = angular_offsets.cos(), angular_offsets.sin()
    else:
        cos_offsets = [math.cos(angular_offsets)] * grids.size()[0]
        sin_offsets = [math.sin(angular_offsets)] * grids.size()[0]

    for roi_center, roi_radii, grid, cos_offset, sin_offset in zip(roi_centers, rois_radii, grids,
                                                                   cos_offsets, sin_offsets):
        normalised_dest_indices = dest_indices - roi_center
        normalised_dest_indices[..., 0], normalised_dest_indices[..., 1] = (
            cos_offset * normalised_dest_indices[..., 0] + sin_offset * normalised_dest_indices[..., 1],
            cos_offset * normalised_dest_indices[..., 1] - sin_offset * normalised_dest_indices[..., 0])
        normalised_dest_indices /= roi_radii
        radii = normalised_dest_indices.norm(dim=-1)
        warped_radii = torch.tanh(radii)
        orientation_x = normalised_dest_indices[..., 0] / radii.clamp(min=1e-9)
        orientation_y = normalised_dest_indices[..., 1] / radii.clamp(min=1e-9)
        grid[..., 0] = ((orientation_x * warped_radii + 1.0) * warped_width / 2.0 -
                        0.5) / (warped_width - 1.0) * 2.0 - 1.0
        grid[..., 1] = ((orientation_y * warped_radii + 1.0) * warped_height / 2.0 -
                        0.5) / (warped_height - 1.0) * 2.0 - 1.0

    return tf.grid_sample(warped_images, grids, mode=interpolation, padding_mode=padding, align_corners=True)

def get_warp_func(polar: int):
    if polar == 0:
        return roi_tanh_warp
    elif polar == 1:
        return roi_tanh_circular_warp
    else:
        return roi_tanh_polar_warp

def get_restore_func(polar: int):
    if polar == 0:
        return roi_tanh_restore
    elif polar == 1:
        return roi_tanh_circular_restore
    else:
        return roi_tanh_polar_restore
