import torch
import torch.nn.functional as tf
from typing import Tuple


# __all__ = ['roi_tanh_warp', 'roi_tanh_restore',
#            'roi_tanh_polor_warp', 'roi_tanh_polar_restore',
#            'roi_tanh_circular_warp', 'roi_tanh_circular_restore']

__all__ = ['roi_tanh_warp', 'roi_tanh_restore']


def arctanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


def roi_tanh_warp(images: torch.Tensor, rois: torch.Tensor, target_size: Tuple[int, int],
                  interpolation: str = 'bilinear', padding: str = 'zeros') -> torch.Tensor:
    image_height, image_width = images.size()[-2:]

    grids = torch.zeros(images.size()[:1] + target_size + (2,),
                        dtype=images.dtype, device=images.device)
    dest_x_indices = torch.arange(target_size[1], dtype=grids.dtype, device=grids.device)
    dest_y_indices = torch.arange(target_size[0], dtype=grids.dtype, device=grids.device)
    for roi, grid in zip(rois, grids):
        roi_center = [(roi[2] + roi[0]) / 2.0, (roi[3] + roi[1]) / 2.0]
        half_roi_size = [(roi[2] - roi[0]) / 2.0, (roi[3] - roi[1]) / 2.0]

        src_x_indices = roi_center[0] + half_roi_size[0] * arctanh(
            (dest_x_indices - target_size[1] / 2.0 + 0.5) / target_size[1] * 2.0)
        grid[..., 0] = (src_x_indices / (image_width - 1) * 2.0 - 1.0).unsqueeze(0).expand_as(grid[..., 0])
        src_y_indices = roi_center[1] + half_roi_size[1] * arctanh(
            (dest_y_indices - target_size[0] / 2.0 + 0.5) / target_size[0] * 2.0)
        grid[..., 1] = (src_y_indices / (image_height - 1) * 2.0 - 1.0).unsqueeze(-1).expand_as(grid[..., 1])

    return tf.grid_sample(images, grids, mode=interpolation, padding_mode=padding)


def roi_tanh_restore(warped_images: torch.Tensor, rois: torch.Tensor, image_size: Tuple[int, int],
                     interpolation: str = 'bilinear', padding: str = 'zeros') -> torch.Tensor:
    warped_height, warped_width = warped_images.size()[-2:]

    grids = torch.zeros(warped_images.size()[:1] + image_size + (2,),
                        dtype=warped_images.dtype, device=warped_images.device)
    dest_x_indices = torch.arange(image_size[1], dtype=grids.dtype, device=grids.device)
    dest_y_indices = torch.arange(image_size[0], dtype=grids.dtype, device=grids.device)
    for roi, grid in zip(rois, grids):
        roi_center = [(roi[2] + roi[0]) / 2.0, (roi[3] + roi[1]) / 2.0]
        half_roi_size = [(roi[2] - roi[0]) / 2.0, (roi[3] - roi[1]) / 2.0]
        src_x_indices = (torch.tanh((dest_x_indices - roi_center[0]) / half_roi_size[0])) - 1.0 / (warped_width-1)

        grid[..., 0] = src_x_indices.unsqueeze(0).expand_as(grid[..., 0])

        src_y_indices = (torch.tanh((dest_y_indices - roi_center[1]) / half_roi_size[1])) - 1.0 / (warped_height-1)

        grid[..., 1] = src_y_indices.unsqueeze(-1).expand_as(grid[..., 1])

    return tf.grid_sample(warped_images, grids, mode=interpolation, padding_mode=padding)

# def roi_tanh_restore(warped_image, roi, image_size, interpolation=cv2.INTER_LINEAR,
#                      border_mode=cv2.BORDER_CONSTANT, border_value=0):
#     warped_size = warped_image.shape[1::-1]
#     roi_center = [(roi[0] + roi[2]) / 2.0, (roi[1] + roi[3]) / 2.0]
#     half_roi_size = [(roi[2] - roi[0]) / 2.0, (roi[3] - roi[1]) / 2.0]
#
#     src_x_indices = np.tile(
#         (np.tanh((np.arange(image_size[1]) - roi_center[0]) /
#                  half_roi_size[0])) / warped_size[0] - 0.5,
#         (image_size[0], 1))
#     src_y_indices = np.tile(
#         (np.tanh((np.arange(image_size[0]) - roi_center[1]) /
#                  half_roi_size[1]) + 1.0) / 2.0 * warped_size[1] - 0.5,
#         (image_size[1], 1)).transpose()
#
#     return cv2.remap(warped_image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
#                      interpolation, borderMode=border_mode, borderValue=border_value)
#
#
# def roi_tanh_polor_warp(image, roi, target_size, angular_offset=0.0, interpolation=cv2.INTER_LINEAR,
#                         border_mode=cv2.BORDER_CONSTANT, border_value=0):
#     roi_center = [(roi[0] + roi[2]) / 2.0, (roi[1] + roi[3]) / 2.0]
#     target_size = np.array(target_size)
#     roi_radii = [(roi[2] - roi[0]) / np.pi ** 0.5, (roi[3] - roi[1]) / np.pi ** 0.5]
#
#     normalised_dest_indices = np.stack(np.meshgrid(np.arange(0.0, 1.0, 1.0 / target_size[1]),
#                                                    np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / target_size[0]) +
#                                                    angular_offset), axis=-1)
#     radii = normalised_dest_indices[..., 0]
#     orientation_x = np.cos(normalised_dest_indices[..., 1])
#     orientation_y = np.sin(normalised_dest_indices[..., 1])
#
#     src_radii = np.arctanh(np.clip(radii, None, 1.0 - 1e-9))
#     src_x_indices = roi_center[0] + roi_radii[0] * src_radii * orientation_x
#     src_y_indices = roi_center[1] + roi_radii[1] * src_radii * orientation_y
#
#     return cv2.remap(image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
#                      interpolation, borderMode=border_mode, borderValue=border_value)
#
#
# def roi_tanh_polar_restore(warped_image, roi, image_size, angular_offset=0.0, interpolation=cv2.INTER_LINEAR,
#                            border_mode=cv2.BORDER_CONSTANT, border_value=0):
#     warped_size = warped_image.shape[1::-1]
#     roi_center = [(roi[0] + roi[2]) / 2.0, (roi[1] + roi[3]) / 2.0]
#     roi_radii = np.array([(roi[2] - roi[0]) / np.pi ** 0.5, (roi[3] - roi[1]) / np.pi ** 0.5])
#
#     dest_indices = np.stack(np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0])),
#                             axis=-1).astype(float)
#     normalised_dest_indices = (dest_indices - np.array(roi_center)) / roi_radii
#     radii = np.linalg.norm(normalised_dest_indices, axis=-1)
#
#     src_radii = np.tanh(radii)
#     src_x_indices = src_radii * warped_size[0]
#     src_y_indices = ((np.arctan2(normalised_dest_indices[..., 1], normalised_dest_indices[..., 0]) -
#                       angular_offset) / 2.0 / np.pi) * warped_size[1] % warped_size[1]
#
#     return cv2.remap(warped_image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
#                      interpolation, borderMode=border_mode, borderValue=border_value)
#
#
# def roi_tanh_circular_warp(image, roi, target_size, interpolation=cv2.INTER_LINEAR,
#                            border_mode=cv2.BORDER_CONSTANT, border_value=0):
#     roi_center = [(roi[0] + roi[2]) / 2.0, (roi[1] + roi[3]) / 2.0]
#     target_size = np.array(target_size)
#     roi_radii = [(roi[2] - roi[0]) / np.pi ** 0.5, (roi[3] - roi[1]) / np.pi ** 0.5]
#
#     dest_indices = np.stack(np.meshgrid(np.arange(target_size[1]), np.arange(target_size[0])), axis=-1).astype(float)
#     normalised_dest_indices = (dest_indices + 0.5 - target_size[::-1] / 2.0) / target_size[::-1] * 2.0
#     radii = np.linalg.norm(normalised_dest_indices, axis=-1)
#     orientation_x = normalised_dest_indices[..., 0] / np.clip(radii, 1e-9, None)
#     orientation_y = normalised_dest_indices[..., 1] / np.clip(radii, 1e-9, None)
#
#     src_radii = np.arctanh(np.clip(radii, None, 1.0 - 1e-9))
#     src_x_indices = roi_center[0] + roi_radii[0] * src_radii * orientation_x
#     src_y_indices = roi_center[1] + roi_radii[1] * src_radii * orientation_y
#
#     return cv2.remap(image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
#                      interpolation, borderMode=border_mode, borderValue=border_value)
#
#
# def roi_tanh_circular_restore(warped_image, roi, image_size, interpolation=cv2.INTER_LINEAR,
#                               border_mode=cv2.BORDER_CONSTANT, border_value=0):
#     warped_size = warped_image.shape[1::-1]
#     roi_center = [(roi[0] + roi[2]) / 2.0, (roi[1] + roi[3]) / 2.0]
#     roi_radii = np.array([(roi[2] - roi[0]) / np.pi ** 0.5, (roi[3] - roi[1]) / np.pi ** 0.5])
#
#     dest_indices = np.stack(np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0])), axis=-1).astype(float)
#     normalised_dest_indices = (dest_indices - np.array(roi_center)) / roi_radii
#     radii = np.linalg.norm(normalised_dest_indices, axis=-1)
#
#     src_radii = np.tanh(radii)
#     orientation_x = normalised_dest_indices[..., 0] / np.clip(radii, 1e-9, None)
#     orientation_y = normalised_dest_indices[..., 1] / np.clip(radii, 1e-9, None)
#
#     src_x_indices = (orientation_x * src_radii + 1.0) * warped_size[0] / 2.0 - 0.5
#     src_y_indices = (orientation_y * src_radii + 1.0) * warped_size[1] / 2.0 - 0.5
#
#     return cv2.remap(warped_image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
#                      interpolation, borderMode=border_mode, borderValue=border_value)