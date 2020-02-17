import os
import cv2
import torch
import numpy as np
from argparse import ArgumentParser
from saic_vision.object_detector import S3FMobileV2Detector


def warp_image(image, face_box, target_size, polar, interpolation=cv2.INTER_LINEAR,
               border_mode=cv2.BORDER_CONSTANT, border_value=0):
    face_center = [(face_box[0] + face_box[2]) / 2.0, (face_box[1] + face_box[3]) / 2.0]
    if polar > 0:
        target_size = np.array(target_size)
        face_radii = [(face_box[2] - face_box[0]) / np.pi ** 0.5, (face_box[3] - face_box[1]) / np.pi ** 0.5]

        if polar > 1:
            normalised_dest_indices = np.stack(np.meshgrid(np.arange(0.0, 1.0, 1.0 / target_size[0]),
                                                           np.arange(-np.pi, np.pi, 2.0 * np.pi / target_size[1])),
                                               axis=-1)
            radii = normalised_dest_indices[..., 0]
            orientation_x = np.cos(normalised_dest_indices[..., 1])
            orientation_y = np.sin(normalised_dest_indices[..., 1])
        else:
            dest_indices = np.stack(np.meshgrid(np.arange(target_size[0]), np.arange(target_size[1])),
                                    axis=-1).astype(float)
            normalised_dest_indices = (dest_indices + 0.5 - target_size / 2.0) / target_size * 2.0
            radii = np.linalg.norm(normalised_dest_indices, axis=-1)
            orientation_x = normalised_dest_indices[..., 0] / np.clip(radii, 1e-9, None)
            orientation_y = normalised_dest_indices[..., 1] / np.clip(radii, 1e-9, None)

        src_radii = np.arctanh(np.clip(radii, None, 1.0 - 1e-9))
        src_x_indices = face_center[0] + face_radii[0] * src_radii * orientation_x
        src_y_indices = face_center[1] + face_radii[1] * src_radii * orientation_y
    else:
        half_face_size = [(face_box[2] - face_box[0]) / 2.0, (face_box[3] - face_box[1]) / 2.0]

        src_x_indices = np.tile(
            face_center[0] + half_face_size[0] * np.arctanh(
                (np.arange(target_size[0]) - target_size[0] / 2.0 + 0.5) / target_size[0] * 2.0),
            (target_size[1], 1))
        src_y_indices = np.tile(
            face_center[1] + half_face_size[1] * np.arctanh(
                (np.arange(target_size[1]) - target_size[1] / 2.0 + 0.5) / target_size[1] * 2.0),
            (target_size[0], 1)).transpose()

    return cv2.remap(image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
                     interpolation, borderMode=border_mode, borderValue=border_value)


def restore_image(warped_image, face_box, image_size, polar, interpolation=cv2.INTER_LINEAR,
                  border_mode=cv2.BORDER_CONSTANT, border_value=0):
    warped_size = warped_image.shape[1::-1]
    face_center = [(face_box[0] + face_box[2]) / 2.0, (face_box[1] + face_box[3]) / 2.0]

    if polar > 0:
        face_radii = np.array([(face_box[2] - face_box[0]) / np.pi ** 0.5,
                               (face_box[3] - face_box[1]) / np.pi ** 0.5])

        dest_indices = np.stack(np.meshgrid(np.arange(image_size[0]), np.arange(image_size[1])),
                                axis=-1).astype(float)
        normalised_dest_indices = (dest_indices - np.array(face_center)) / face_radii
        radii = np.linalg.norm(normalised_dest_indices, axis=-1)

        src_radii = np.tanh(radii)
        if polar > 1:
            src_x_indices = src_radii * warped_size[0]
            src_y_indices = (np.arctan2(normalised_dest_indices[..., 1],
                                        normalised_dest_indices[..., 0]) / 2.0 / np.pi + 0.5) * warped_size[1]
        else:
            orientation_x = normalised_dest_indices[..., 0] / np.clip(radii, 1e-9, None)
            orientation_y = normalised_dest_indices[..., 1] / np.clip(radii, 1e-9, None)

            src_x_indices = (orientation_x * src_radii + 1.0) * warped_size[0] / 2.0 - 0.5
            src_y_indices = (orientation_y * src_radii + 1.0) * warped_size[1] / 2.0 - 0.5
    else:
        half_face_size = [(face_box[2] - face_box[0]) / 2.0, (face_box[3] - face_box[1]) / 2.0]

        src_x_indices = np.tile(
            (np.tanh((np.arange(image_size[0]) - face_center[0]) /
                     half_face_size[0]) + 1.0) / 2.0 * warped_size[0] - 0.5,
            (image_size[1], 1))
        src_y_indices = np.tile(
            (np.tanh((np.arange(image_size[1]) - face_center[1]) /
                     half_face_size[1]) + 1.0) / 2.0 * warped_size[1] - 0.5,
            (image_size[0], 1)).transpose()

    return cv2.remap(warped_image, src_x_indices.astype(np.float32), src_y_indices.astype(np.float32),
                     interpolation, borderMode=border_mode, borderValue=border_value)


def main():
    parser = ArgumentParser()
    parser.add_argument('-v', '--video', help='video source')
    parser.add_argument('-x', '--width', help='face width', type=int, default=256)
    parser.add_argument('-y', '--height', help='face height', type=int, default=256)
    parser.add_argument('-p', '--polar', help='use polar coordinates', type=int, default=0)
    parser.add_argument('-r', '--restore', help='show restored frames', action='store_true')
    args = parser.parse_args()

    # Make the models run a bit faster
    torch.backends.cudnn.benchmark = True

    # Create object detector
    detector = S3FMobileV2Detector(th=0.25, device='cuda:0')
    print('S3FD object detector created.')

    # Open webcam
    if os.path.exists(args.video):
        vid = cv2.VideoCapture(args.video)
        print('Video file opened: %s.' % args.video)
    else:
        vid = cv2.VideoCapture(int(args.video))
        print('Webcam #%d opened.' % int(args.video))

    # Detect objects in the frames
    try:
        frame_number = 0
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        print('Face detection started, press \'Q\' to quit.')
        while True:
            _, frame = vid.read()
            if frame is None:
                break
            else:
                # Face detection
                bboxes, labels, probs = detector.detect_from_image(frame)
                face_boxes = [bboxes[idx] for idx, label in enumerate(labels) if label == 1]
                if len(face_boxes) > 0:
                    biggest_face_idx = int(np.argmax([(bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
                                                      for bbox in face_boxes]))

                    # Warping
                    warped_frame = warp_image(frame, face_boxes[biggest_face_idx], (args.width, args.height),
                                              args.polar, border_mode=cv2.BORDER_REPLICATE)
                    if args.restore:
                        restored_frame = restore_image(warped_frame, face_boxes[biggest_face_idx], frame.shape[1::-1],
                                                       args.polar, border_mode=cv2.BORDER_REPLICATE)
                    else:
                        restored_frame = None

                    # Rendering
                    for idx, bbox in enumerate(face_boxes):
                        if idx == biggest_face_idx:
                            border_colour = (0, 0, 255)
                        else:
                            border_colour = (128, 128, 128)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                      color=border_colour, thickness=2)
                else:
                    warped_frame = None
                    restored_frame = None

                # Show the result
                print('Frame #%d: %d faces(s) detected.' % (frame_number, len(face_boxes)))
                cv2.imshow(script_name, frame)
                if args.restore:
                    if restored_frame is None:
                        cv2.destroyWindow('Restored')
                    else:
                        cv2.imshow('Restored', restored_frame)
                if warped_frame is None:
                    cv2.destroyWindow('Warped')
                else:
                    cv2.imshow('Warped', warped_frame)
                key = cv2.waitKey(1) % 2 ** 16
                if key == ord('q') or key == ord('Q'):
                    print("\'Q\' pressed, we are done here.")
                    break
                else:
                    frame_number += 1
    finally:
        cv2.destroyAllWindows()
        vid.release()
        print('We are done here.')


if __name__ == '__main__':
    main()
