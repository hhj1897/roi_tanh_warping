import os
import cv2
import torch
import numpy as np
from argparse import ArgumentParser
from saic_vision.object_detector import S3FMobileV2Detector

from ibug.roi_tanh_warping import *
from ibug.roi_tanh_warping import reference_impl as ref


def test_reference_impl(frame, face_box, target_size, offset, restore, compare, keep_aspect_ratio, reverse):
    if reverse:
        # ROI-tanh warping
        roi_tanh_frame = ref.roi_tanh_warp(frame, face_box, target_size, offset,
                                           border_mode=cv2.BORDER_REPLICATE)

        # ROI-tanh to ROI-tanh-polar
        roi_tanh_polar_frame = ref.roi_tanh_to_roi_tanh_polar(roi_tanh_frame, face_box,
                                                              border_mode=cv2.BORDER_REPLICATE,
                                                              keep_aspect_ratio=keep_aspect_ratio)

        # Restore from ROI-tanh-polar
        if restore:
            restored_frame = ref.roi_tanh_polar_restore(roi_tanh_polar_frame, face_box, frame.shape[:2], offset,
                                                        border_mode=cv2.BORDER_REPLICATE)
        else:
            restored_frame = None

        # Compute difference
        if compare:
            reference_frame = ref.roi_tanh_polar_warp(frame, face_box, target_size, offset,
                                                      border_mode=cv2.BORDER_REPLICATE,
                                                      keep_aspect_ratio=keep_aspect_ratio)
            diff_frame = np.abs(reference_frame.astype(int) - roi_tanh_polar_frame.astype(int)).astype(np.uint8)
        else:
            diff_frame = None
    else:
        # ROI-tanh-polar warping
        roi_tanh_polar_frame = ref.roi_tanh_polar_warp(frame, face_box, target_size, offset,
                                                       border_mode=cv2.BORDER_REPLICATE,
                                                       keep_aspect_ratio=keep_aspect_ratio)

        # ROI-tanh-polar to ROI-tanh
        roi_tanh_frame = ref.roi_tanh_polar_to_roi_tanh(roi_tanh_polar_frame, face_box,
                                                        border_mode=cv2.BORDER_REPLICATE,
                                                        keep_aspect_ratio=keep_aspect_ratio)

        # Restore from ROI-tanh
        if restore:
            restored_frame = ref.roi_tanh_restore(roi_tanh_frame, face_box, frame.shape[:2], offset,
                                                  border_mode=cv2.BORDER_REPLICATE)
        else:
            restored_frame = None

        # Compute difference
        if compare:
            reference_frame = ref.roi_tanh_warp(frame, face_box, target_size, offset,
                                                border_mode=cv2.BORDER_REPLICATE)
            diff_frame = np.abs(reference_frame.astype(int) - roi_tanh_frame.astype(int)).astype(np.uint8)
        else:
            diff_frame = None

    return roi_tanh_polar_frame, roi_tanh_frame, restored_frame, diff_frame


def main():
    parser = ArgumentParser()
    parser.add_argument('-v', '--video', help='video source')
    parser.add_argument('-x', '--width', help='face width', type=int, default=256)
    parser.add_argument('-y', '--height', help='face height', type=int, default=256)
    parser.add_argument('-o', '--offset', help='angular offset, only used when polar>0', type=float, default=0.0)
    parser.add_argument('-r', '--restore', help='show restored frames', action='store_true')
    parser.add_argument('-c', '--compare', help='compare with directly warped frames', action='store_true')
    parser.add_argument('-k', '--keep-aspect-ratio', help='Keep aspect ratio in tanh-polar or tanh-circular warping',
                        action='store_true')
    parser.add_argument('-i', '--reverse', help='perform computation in the reverse direction', action='store_true')
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

                    # Test the warping functions
                    roi_tanh_polar_frame, roi_tanh_frame, restored_frame, diff_frame = test_reference_impl(
                        frame, face_boxes[biggest_face_idx], (args.height, args.width),
                        args.offset / 180.0 * np.pi, args.restore, args.compare,
                        args.keep_aspect_ratio, args.reverse)

                    # Rendering
                    for idx, bbox in enumerate(face_boxes):
                        if idx == biggest_face_idx:
                            border_colour = (0, 0, 255)
                        else:
                            border_colour = (128, 128, 128)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                      color=border_colour, thickness=2)
                else:
                    roi_tanh_polar_frame = None
                    roi_tanh_frame = None
                    restored_frame = None
                    diff_frame = None

                # Show the result
                print('Frame #%d: %d faces(s) detected.' % (frame_number, len(face_boxes)))
                cv2.imshow(script_name, frame)
                if args.reverse:
                    if roi_tanh_frame is None:
                        cv2.destroyWindow('ROI-Tanh')
                    else:
                        cv2.imshow('ROI-Tanh', roi_tanh_frame)
                    if roi_tanh_polar_frame is None:
                        cv2.destroyWindow('ROI-Tanh-Polar')
                    else:
                        cv2.imshow('ROI-Tanh-Polar', roi_tanh_polar_frame)
                else:
                    if roi_tanh_polar_frame is None:
                        cv2.destroyWindow('ROI-Tanh-Polar')
                    else:
                        cv2.imshow('ROI-Tanh-Polar', roi_tanh_polar_frame)
                    if roi_tanh_frame is None:
                        cv2.destroyWindow('ROI-Tanh')
                    else:
                        cv2.imshow('ROI-Tanh', roi_tanh_frame)
                if args.restore:
                    if restored_frame is None:
                        cv2.destroyWindow('Restored')
                    else:
                        cv2.imshow('Restored', restored_frame)
                if args.compare:
                    if diff_frame is None:
                        cv2.destroyWindow('Difference')
                    else:
                        cv2.imshow('Difference', diff_frame)
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
