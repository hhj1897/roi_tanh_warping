import cv2
import numpy as np


__all__ = ['roi_tanh_warp', 'roi_tanh_restore',
           'roi_tanh_polar_warp', 'roi_tanh_polar_restore',
           'roi_tanh_circular_warp', 'roi_tanh_circular_restore']


def roi_tanh_warp(image, roi, target_size, angular_offset: float = 0.0,
                  interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=0):
    roi_center = [(roi[0] + roi[2]) / 2.0, (roi[1] + roi[3]) / 2.0]
    half_roi_size = [(roi[2] - roi[0]) / 2.0, (roi[3] - roi[1]) / 2.0]

    normalised_dest_x_indices = np.arange(-1.0, 1.0, 2.0 / target_size[1]) + 1.0 / target_size[1]
    normalised_dest_y_indices = np.arange(-1.0, 1.0, 2.0 / target_size[0]) + 1.0 / target_size[0]

    src_x_indices = np.tile(roi_center[0] + half_roi_size[0] * np.arctanh(normalised_dest_x_indices),
                            (target_size[0], 1))
    src_y_indices = np.tile(roi_center[1] + half_roi_size[1] * np.arctanh(normalised_dest_y_indices),
                            (target_size[1], 1)).transpose()

    return cv2.remap(image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
                     interpolation, borderMode=border_mode, borderValue=border_value)


def roi_tanh_restore(warped_image, roi, image_size, angular_offset: float = 0.0,
                     interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=0):
    warped_height, warped_width = warped_image.shape[:2]
    roi_center = [(roi[0] + roi[2]) / 2.0, (roi[1] + roi[3]) / 2.0]
    half_roi_size = [(roi[2] - roi[0]) / 2.0, (roi[3] - roi[1]) / 2.0]

    src_x_indices = np.tile((np.tanh((np.arange(image_size[1]) - roi_center[0]) / half_roi_size[0]) +
                             1.0) / 2.0 * warped_width - 0.5,
                            (image_size[0], 1))
    src_y_indices = np.tile((np.tanh((np.arange(image_size[0]) - roi_center[1]) / half_roi_size[1]) +
                             1.0) / 2.0 * warped_height - 0.5, (image_size[1], 1)).transpose()

    return cv2.remap(warped_image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
                     interpolation, borderMode=border_mode, borderValue=border_value)


def roi_tanh_polar_warp(image, roi, target_size, angular_offset=0.0,
                        interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=0):
    roi_center = [(roi[0] + roi[2]) / 2.0, (roi[1] + roi[3]) / 2.0]
    roi_radii = [(roi[2] - roi[0]) / np.pi ** 0.5, (roi[3] - roi[1]) / np.pi ** 0.5]

    normalised_dest_indices = np.stack(np.meshgrid(np.arange(0.0, 1.0, 1.0 / target_size[1]),
                                                   np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / target_size[0]) +
                                                   angular_offset), axis=-1)
    radii = normalised_dest_indices[..., 0]
    orientation_x = np.cos(normalised_dest_indices[..., 1])
    orientation_y = np.sin(normalised_dest_indices[..., 1])

    src_radii = np.arctanh(radii)
    src_x_indices = roi_center[0] + roi_radii[0] * src_radii * orientation_x
    src_y_indices = roi_center[1] + roi_radii[1] * src_radii * orientation_y

    return cv2.remap(image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
                     interpolation, borderMode=border_mode, borderValue=border_value)


def roi_tanh_polar_restore(warped_image, roi, image_size, angular_offset=0.0,
                           interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=0):
    warped_height, warped_width = warped_image.shape[:2]
    roi_center = np.array([(roi[0] + roi[2]) / 2.0, (roi[1] + roi[3]) / 2.0])
    roi_radii = np.array([(roi[2] - roi[0]) / np.pi ** 0.5, (roi[3] - roi[1]) / np.pi ** 0.5])

    dest_indices = np.stack(np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0])),
                            axis=-1).astype(float)
    normalised_dest_indices = (dest_indices - roi_center) / roi_radii
    radii = np.linalg.norm(normalised_dest_indices, axis=-1)

    src_radii = np.tanh(radii)
    src_x_indices = src_radii * warped_width
    src_y_indices = np.mod(((np.arctan2(normalised_dest_indices[..., 1], normalised_dest_indices[..., 0]) -
                             angular_offset) / 2.0 / np.pi) * warped_height, warped_height)

    return cv2.remap(warped_image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
                     interpolation, borderMode=border_mode, borderValue=border_value)


def roi_tanh_circular_warp(image, roi, target_size, angular_offset: float = 0.0,
                           interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=0):
    roi_center = [(roi[0] + roi[2]) / 2.0, (roi[1] + roi[3]) / 2.0]
    roi_radii = [(roi[2] - roi[0]) / np.pi ** 0.5, (roi[3] - roi[1]) / np.pi ** 0.5]
    cos_offset, sin_offset = np.cos(angular_offset), np.sin(angular_offset)

    normalised_dest_x_indices = np.arange(-1.0, 1.0, 2.0 / target_size[1]) + 1.0 / target_size[1]
    normalised_dest_y_indices = np.arange(-1.0, 1.0, 2.0 / target_size[0]) + 1.0 / target_size[0]
    normalised_dest_indices = np.stack(np.meshgrid(normalised_dest_x_indices, normalised_dest_y_indices), axis=-1)
    radii = np.linalg.norm(normalised_dest_indices, axis=-1)
    orientation_x = normalised_dest_indices[..., 0] / np.clip(radii, 1e-9, None)
    orientation_y = normalised_dest_indices[..., 1] / np.clip(radii, 1e-9, None)
    orientation_x, orientation_y = (cos_offset * orientation_x - sin_offset * orientation_y,
                                    cos_offset * orientation_y + sin_offset * orientation_x)

    src_radii = np.arctanh(np.clip(radii, None, 1.0 - 1e-9))
    src_x_indices = roi_center[0] + roi_radii[0] * src_radii * orientation_x
    src_y_indices = roi_center[1] + roi_radii[1] * src_radii * orientation_y

    return cv2.remap(image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
                     interpolation, borderMode=border_mode, borderValue=border_value)


def roi_tanh_circular_restore(warped_image, roi, image_size, angular_offset: float = 0.0,
                              interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=0):
    warped_height, warped_width = warped_image.shape[:2]
    roi_center = np.array([(roi[0] + roi[2]) / 2.0, (roi[1] + roi[3]) / 2.0])
    roi_radii = np.array([(roi[2] - roi[0]) / np.pi ** 0.5, (roi[3] - roi[1]) / np.pi ** 0.5])
    cos_offset, sin_offset = np.cos(angular_offset), np.sin(angular_offset)

    dest_indices = np.stack(np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0])), axis=-1).astype(float)
    normalised_dest_indices = (dest_indices - roi_center) / roi_radii
    radii = np.linalg.norm(normalised_dest_indices, axis=-1)

    src_radii = np.tanh(radii)
    orientation_x = normalised_dest_indices[..., 0] / np.clip(radii, 1e-9, None)
    orientation_y = normalised_dest_indices[..., 1] / np.clip(radii, 1e-9, None)
    orientation_x, orientation_y = (cos_offset * orientation_x + sin_offset * orientation_y,
                                    cos_offset * orientation_y - sin_offset * orientation_x)

    src_x_indices = (orientation_x * src_radii + 1.0) * warped_width / 2.0 - 0.5
    src_y_indices = (orientation_y * src_radii + 1.0) * warped_height / 2.0 - 0.5

    return cv2.remap(warped_image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
                     interpolation, borderMode=border_mode, borderValue=border_value)
