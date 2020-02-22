import math
import torch
import torch.nn.functional as tf
from typing import Tuple


__all__ = ['roi_tanh_warp', 'roi_tanh_restore',
           'roi_tanh_polar_warp', 'roi_tanh_polar_restore',
           'roi_tanh_circular_warp', 'roi_tanh_circular_restore']


def arctanh(x):
    return torch.log((1.0 + x) / (1.0 - x)) / 2.0


def roi_tanh_warp(images: torch.Tensor, rois: torch.Tensor, target_size: Tuple[int, int],
                  interpolation: str = 'bilinear', padding: str = 'zeros') -> torch.Tensor:
    image_height, image_width = images.size()[-2:]
    roi_centers = (rois[:, 2:4] + rois[:, :2]) / 2.0
    half_roi_sizes = (rois[:, 2:4] - rois[:, :2]) / 2.0

    grids = torch.zeros(images.size()[:1] + target_size + (2,), dtype=images.dtype, device=images.device)
    wapred_x_indices = arctanh(torch.arange(-1.0, 1.0, 2.0 / target_size[1], dtype=grids.dtype,
                                                 device=grids.device) + 1.0 / target_size[1])
    wapred_y_indices = arctanh(torch.arange(-1.0, 1.0, 2.0 / target_size[0], dtype=grids.dtype,
                                                 device=grids.device) + 1.0 / target_size[0])
    for roi_center, half_roi_size, grid in zip(roi_centers, half_roi_sizes, grids):
        src_x_indices = roi_center[0] + half_roi_size[0] * wapred_x_indices
        grid[..., 0] = (src_x_indices / (image_width - 1.0) * 2.0 - 1.0).unsqueeze(0).expand_as(grid[..., 0])
        src_y_indices = roi_center[1] + half_roi_size[1] * wapred_y_indices
        grid[..., 1] = (src_y_indices / (image_height - 1.0) * 2.0 - 1.0).unsqueeze(-1).expand_as(grid[..., 1])

    return tf.grid_sample(images, grids, mode=interpolation, padding_mode=padding, align_corners=True)


def roi_tanh_restore(warped_images: torch.Tensor, rois: torch.Tensor, image_size: Tuple[int, int],
                     interpolation: str = 'bilinear', padding: str = 'zeros') -> torch.Tensor:
    warped_height, warped_width = warped_images.size()[-2:]
    roi_centers = (rois[:, 2:4] + rois[:, :2]) / 2.0
    half_roi_sizes = (rois[:, 2:4] - rois[:, :2]) / 2.0

    grids = torch.zeros(warped_images.size()[:1] + image_size + (2,),
                        dtype=warped_images.dtype, device=warped_images.device)
    dest_x_indices = torch.arange(image_size[1], dtype=grids.dtype, device=grids.device)
    dest_y_indices = torch.arange(image_size[0], dtype=grids.dtype, device=grids.device)
    for roi_center, half_roi_size, grid in zip(roi_centers, half_roi_sizes, grids):
        src_x_indices = (torch.tanh((dest_x_indices - roi_center[0]) / half_roi_size[0]) +
                         1.0) / 2.0 * warped_width - 0.5
        grid[..., 0] = (src_x_indices / (warped_width - 1.0) * 2.0 - 1.0).unsqueeze(0).expand_as(grid[..., 0])
        src_y_indices = (torch.tanh((dest_y_indices - roi_center[1]) / half_roi_size[1]) +
                         1.0) / 2.0 * warped_height - 0.5
        grid[..., 1] = (src_y_indices / (warped_height - 1.0) * 2.0 - 1.0).unsqueeze(-1).expand_as(grid[..., 1])

    return tf.grid_sample(warped_images, grids, mode=interpolation, padding_mode=padding, align_corners=True)


def roi_tanh_polar_warp(images: torch.Tensor, rois: torch.Tensor, target_size: Tuple[int, int],
                        angular_offset: float=0.0, interpolation: str = 'bilinear',
                        padding: str = 'zeros') -> torch.Tensor:
    image_height, image_width = images.size()[-2:]
    roi_centers = (rois[:, 2:4] + rois[:, :2]) / 2.0
    rois_radii = (rois[:, 2:4] - rois[:, :2]) / math.pi ** 0.5

    grids = torch.zeros(images.size()[:1] + target_size + (2,), dtype=images.dtype, device=images.device)
    wapred_radii = arctanh(torch.arange(0.0, 1.0, 1.0 / target_size[1], dtype=grids.dtype,
                                        device=grids.device)).unsqueeze(0).expand(target_size)
    thetas = torch.arange(0.0, 2.0 * math.pi, 2.0 * math.pi / target_size[0],
                          dtype=grids.dtype, device=grids.device) + angular_offset
    warped_x_indices = wapred_radii * torch.cos(thetas).unsqueeze(-1).expand(target_size)
    warped_y_indices = wapred_radii * torch.sin(thetas).unsqueeze(-1).expand(target_size)
    for roi_center, roi_radii, grid in zip(roi_centers, rois_radii, grids):
        grid[..., 0] = (roi_center[0] + roi_radii[0] * warped_x_indices) / (image_width - 1.0) * 2.0 - 1.0
        grid[..., 1] = (roi_center[1] + roi_radii[1] * warped_y_indices) / (image_height - 1.0) * 2.0 - 1.0

    return tf.grid_sample(images, grids, mode=interpolation, padding_mode=padding, align_corners=True)


def roi_tanh_polar_restore(warped_images: torch.Tensor, rois: torch.Tensor, image_size: Tuple[int, int],
                           angular_offset: float=0.0, interpolation: str = 'bilinear',
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
    for roi_center, roi_radii, grid in zip(roi_centers, rois_radii, grids):
        normalised_dest_indices = (dest_indices - roi_center) / roi_radii
        radii = normalised_dest_indices.norm(dim=-1)
        grid[..., 0] = torch.tanh(radii) * 2.0 * warped_width / (warped_width - 1.0) - 1.0
        grid[..., 1] = ((torch.atan2(normalised_dest_indices[..., 1], normalised_dest_indices[..., 0]) -
                         angular_offset) / math.pi).remainder(2.0) * warped_height / (warped_height - 1.0) - 1.0

    return tf.grid_sample(warped_images, grids, mode=interpolation, padding_mode=padding, align_corners=True)


def roi_tanh_circular_warp(images: torch.Tensor, rois: torch.Tensor, target_size: Tuple[int, int],
                           interpolation: str = 'bilinear', padding: str = 'zeros') -> torch.Tensor:
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
    warped_radii = arctanh(radii.clamp(max=1.0 - 1e-9))
    warped_x_indices = warped_radii * orientation_x
    warped_y_indices = warped_radii * orientation_y
    for roi_center, roi_radii, grid in zip(roi_centers, rois_radii, grids):
        grid[..., 0] = (roi_center[0] + roi_radii[0] * warped_x_indices) / (image_width - 1.0) * 2.0 - 1.0
        grid[..., 1] = (roi_center[1] + roi_radii[1] * warped_y_indices) / (image_height - 1.0) * 2.0 - 1.0

    return tf.grid_sample(images, grids, mode=interpolation, padding_mode=padding, align_corners=True)


def roi_tanh_circular_restore(warped_images: torch.Tensor, rois: torch.Tensor, image_size: Tuple[int, int],
                              interpolation: str = 'bilinear', padding: str = 'zeros') -> torch.Tensor:
    warped_height, warped_width = warped_images.size()[-2:]
    roi_centers = (rois[:, 2:4] + rois[:, :2]) / 2.0
    rois_radii = (rois[:, 2:4] - rois[:, :2]) / math.pi ** 0.5

    grids = torch.zeros(warped_images.size()[:1] + image_size + (2,),
                        dtype=warped_images.dtype, device=warped_images.device)
    dest_x_indices = torch.arange(image_size[1], dtype=warped_images.dtype, device=warped_images.device)
    dest_y_indices = torch.arange(image_size[0], dtype=warped_images.dtype, device=warped_images.device)
    dest_indices = torch.cat((dest_x_indices.unsqueeze(0).expand(image_size).unsqueeze(-1),
                              dest_y_indices.unsqueeze(-1).expand(image_size).unsqueeze(-1)), -1)
    for roi_center, roi_radii, grid in zip(roi_centers, rois_radii, grids):
        normalised_dest_indices = (dest_indices - roi_center) / roi_radii
        radii = normalised_dest_indices.norm(dim=-1)
        warped_radii = torch.tanh(radii)
        orientation_x = normalised_dest_indices[..., 0] / radii.clamp(min=1e-9)
        orientation_y = normalised_dest_indices[..., 1] / radii.clamp(min=1e-9)
        grid[..., 0] = ((orientation_x * warped_radii + 1.0) * warped_width / 2.0 -
                        0.5) / (warped_width - 1.0) * 2.0 - 1.0
        grid[..., 1] = ((orientation_y * warped_radii + 1.0) * warped_height / 2.0 -
                        0.5) / (warped_height - 1.0) * 2.0 - 1.0

    return tf.grid_sample(warped_images, grids, mode=interpolation, padding_mode=padding, align_corners=True)
