import os
import cv2
import torch
import numpy as np
from argparse import ArgumentParser
from saic_vision.object_detector import S3FMobileV2Detector

from ibug.roi_tanh_warping import *
from ibug.roi_tanh_warping import reference_impl as ref


def test_pytorch_impl(frame, face_box, target_size, polar, offset, restore, square, nearest, keep_aspect_ratio):
    # Preparation
    frames = torch.from_numpy(frame.astype(np.float32)).to(torch.device('cuda:0')).permute(2, 0, 1).unsqueeze(0)
    face_boxes = torch.from_numpy(np.array(face_box[:4], dtype=np.float32)).to(frames.device).unsqueeze(0)
    if square:
        face_boxes = make_square_rois(face_boxes)

    # Warping
    if polar > 1:
        warped_frames = roi_tanh_circular_warp(frames, face_boxes, target_size, angular_offsets=offset,
                                               padding='border', keep_aspect_ratio=keep_aspect_ratio)
    elif polar > 0:
        warped_frames = roi_tanh_polar_warp(frames, face_boxes, target_size, angular_offsets=offset,
                                            padding='border', keep_aspect_ratio=keep_aspect_ratio)
    else:
        warped_frames = roi_tanh_warp(frames, face_boxes, target_size, angular_offsets=offset, padding='border')
    warped_frame = warped_frames[0].detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    # Restoration
    interpolation = 'nearest' if nearest else 'bilinear'
    if restore:
        if polar > 1:
            restored_frames = roi_tanh_circular_restore(warped_frames, face_boxes, frames.size()[-2:],
                                                        angular_offsets=offset, interpolation=interpolation,
                                                        padding='border', keep_aspect_ratio=keep_aspect_ratio)
        elif polar > 0:
            restored_frames = roi_tanh_polar_restore(warped_frames, face_boxes, frames.size()[-2:],
                                                     angular_offsets=offset, interpolation=interpolation,
                                                     padding='border', keep_aspect_ratio=keep_aspect_ratio)
        else:
            restored_frames = roi_tanh_restore(warped_frames, face_boxes, frames.size()[-2:],
                                               angular_offsets=offset, interpolation=interpolation,
                                               padding='border')
        restored_frame = restored_frames[0].detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    else:
        restored_frame = None

    return warped_frame, restored_frame


def test_reference_impl(frame, face_box, target_size, polar, offset, restore, square, nearest, keep_aspect_ratio):
    # Preparation
    if square:
        face_box = ref.make_square_rois(np.array(face_box))

    # Warping
    if polar > 1:
        warped_frame = ref.roi_tanh_circular_warp(frame, face_box, target_size, angular_offset=offset,
                                                  border_mode=cv2.BORDER_REPLICATE,
                                                  keep_aspect_ratio=keep_aspect_ratio)
    elif polar > 0:
        warped_frame = ref.roi_tanh_polar_warp(frame, face_box, target_size, angular_offset=offset,
                                               border_mode=cv2.BORDER_REPLICATE,
                                               keep_aspect_ratio=keep_aspect_ratio)
    else:
        warped_frame = ref.roi_tanh_warp(frame, face_box, target_size, angular_offset=offset,
                                         border_mode=cv2.BORDER_REPLICATE)

    # Restoration
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    if restore:
        if polar > 1:
            restored_frame = ref.roi_tanh_circular_restore(warped_frame, face_box, frame.shape[:2],
                                                           angular_offset=offset, interpolation=interpolation,
                                                           border_mode=cv2.BORDER_REPLICATE,
                                                           keep_aspect_ratio=keep_aspect_ratio)
        elif polar > 0:
            restored_frame = ref.roi_tanh_polar_restore(warped_frame, face_box, frame.shape[:2],
                                                        angular_offset=offset, interpolation=interpolation,
                                                        border_mode=cv2.BORDER_REPLICATE,
                                                        keep_aspect_ratio=keep_aspect_ratio)
        else:
            restored_frame = ref.roi_tanh_restore(warped_frame, face_box, frame.shape[:2],
                                                  angular_offset=offset, interpolation=interpolation,
                                                  border_mode=cv2.BORDER_REPLICATE)
    else:
        restored_frame = None

    return warped_frame, restored_frame


def main():
    parser = ArgumentParser()
    parser.add_argument('-v', '--video', help='video source')
    parser.add_argument('-x', '--width', help='face width', type=int, default=256)
    parser.add_argument('-y', '--height', help='face height', type=int, default=256)
    parser.add_argument('-p', '--polar', help='use polar coordinates', type=int, default=0)
    parser.add_argument('-o', '--offset', help='angular offset, only used when polar>0', type=float, default=0.0)
    parser.add_argument('-r', '--restore', help='show restored frames', action='store_true')
    parser.add_argument('-c', '--compare', help='compare with reference implementation', action='store_true')
    parser.add_argument('-s', '--square', help='use square-shaped detection box', action='store_true')
    parser.add_argument('-n', '--nearest', help='use nearest-neighbour interpolation during restoration',
                        action='store_true')
    parser.add_argument('-k', '--keep-aspect-ratio', help='Keep aspect ratio in tanh-polar or tanh-circular warping',
                        action='store_true')
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

                    warped_frame, restored_frame = test_pytorch_impl(frame, face_boxes[biggest_face_idx],
                                                                     (args.height, args.width), args.polar,
                                                                     args.offset / 180.0 * np.pi, args.restore,
                                                                     args.square, args.nearest,
                                                                     args.keep_aspect_ratio)
                    if args.compare:
                        ref_warped_frame, ref_restored_frame = test_reference_impl(frame,
                                                                                   face_boxes[biggest_face_idx],
                                                                                   (args.height, args.width),
                                                                                   args.polar,
                                                                                   args.offset / 180.0 * np.pi,
                                                                                   args.restore, args.square,
                                                                                   args.nearest,
                                                                                   args.keep_aspect_ratio)
                        diff_warped_frame = np.abs(ref_warped_frame.astype(int) -
                                                   warped_frame.astype(int)).astype(np.uint8)
                        if args.restore:
                            diff_restored_frame = np.abs(ref_restored_frame.astype(int) -
                                                         restored_frame.astype(int)).astype(np.uint8)
                        else:
                            diff_restored_frame = None
                    else:
                        diff_warped_frame = None
                        diff_restored_frame = None

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
                    diff_warped_frame = None
                    diff_restored_frame = None

                # Show the result
                print('Frame #%d: %d faces(s) detected.' % (frame_number, len(face_boxes)))
                cv2.imshow(script_name, frame)
                if args.compare:
                    if args.restore:
                        if diff_restored_frame is None:
                            cv2.destroyWindow('Restored (diff)')
                        else:
                            cv2.imshow('Restored (diff)', diff_restored_frame)
                    if diff_warped_frame is None:
                        cv2.destroyWindow('Warped (diff)')
                    else:
                        cv2.imshow('Warped (diff)', diff_warped_frame)
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
