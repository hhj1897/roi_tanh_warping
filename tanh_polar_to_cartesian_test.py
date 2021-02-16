import os
import cv2
import time
import torch
import numpy as np
from typing import Tuple, Optional
from argparse import ArgumentParser
from ibug.face_detection import RetinaFacePredictor

from ibug.roi_tanh_warping import *
from ibug.roi_tanh_warping import reference_impl as ref


@torch.no_grad()
def test_pytorch_impl(device: str, frame: np.ndarray, face_box: np.ndarray, target_width: int, target_height: int,
                      offset: float, restore: bool, compare: bool, compare_direct: bool, square: bool,
                      keep_aspect_ratio: bool, reverse: bool) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray],
                                                                       Optional[np.ndarray], Optional[np.ndarray]]:
    # Preparation
    if square:
        face_box = ref.make_square_rois(face_box[:4])
    frames = torch.from_numpy(frame.astype(np.float32)).to(device).permute(2, 0, 1).unsqueeze(0)
    face_boxes = torch.from_numpy(face_box[:4]).to(device).unsqueeze(0)

    if reverse:
        # ROI-tanh warping
        roi_tanh_frames = roi_tanh_warp(frames, face_boxes, target_width, target_height, offset, padding='border')

        # ROI-tanh to ROI-tanh-polar
        roi_tanh_polar_frames = roi_tanh_to_roi_tanh_polar(roi_tanh_frames, face_boxes, padding='border',
                                                           keep_aspect_ratio=keep_aspect_ratio)

        # Restore from ROI-tanh-polar
        if restore:
            restored_frames = roi_tanh_polar_restore(roi_tanh_polar_frames, face_boxes, *frame.shape[1::-1],
                                                     angular_offsets=offset, padding='border',
                                                     keep_aspect_ratio=keep_aspect_ratio)
        else:
            restored_frames = None

        # Compute difference with direct warping
        if compare_direct:
            reference_frames = roi_tanh_polar_warp(frames, face_boxes, target_width, target_height, offset,
                                                   padding='border', keep_aspect_ratio=keep_aspect_ratio)
            diff_directs = torch.abs(reference_frames - roi_tanh_polar_frames)
        else:
            diff_directs = None
    else:
        # ROI-tanh-polar warping
        roi_tanh_polar_frames = roi_tanh_polar_warp(frames, face_boxes, target_width, target_height, offset,
                                                    padding='border', keep_aspect_ratio=keep_aspect_ratio)

        # ROI-tanh-polar to ROI-tanh
        roi_tanh_frames = roi_tanh_polar_to_roi_tanh(roi_tanh_polar_frames, face_boxes, padding='border',
                                                     keep_aspect_ratio=keep_aspect_ratio)

        # Restore from ROI-tanh
        if restore:
            restored_frames = roi_tanh_restore(roi_tanh_frames, face_boxes, *frame.shape[1::-1],
                                               angular_offsets=offset, padding='border')
        else:
            restored_frames = None

        # Compute difference with direct warping
        if compare_direct:
            reference_frames = roi_tanh_warp(frames, face_boxes, target_width, target_height, offset, padding='border')
            diff_directs = torch.abs(reference_frames - roi_tanh_frames)
        else:
            diff_directs = None

    roi_tanh_polar_frame = roi_tanh_polar_frames[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    roi_tanh_frame = roi_tanh_frames[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    if restored_frames is None:
        restored_frame = None
    else:
        restored_frame = restored_frames[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    if diff_directs is None:
        diff_direct = None
    else:
        diff_direct = diff_directs[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    if compare:
        if reverse:
            ref_roi_tanh_polar_frame = ref.roi_tanh_to_roi_tanh_polar(
                roi_tanh_frame, face_box, target_width, target_height,
                border_mode=cv2.BORDER_REPLICATE, keep_aspect_ratio=keep_aspect_ratio)
            diff_ref = np.abs(ref_roi_tanh_polar_frame.astype(int) - roi_tanh_polar_frame.astype(int)).astype(np.uint8)
        else:
            ref_roi_tanh_frame = ref.roi_tanh_polar_to_roi_tanh(
                roi_tanh_polar_frame, face_box, target_width, target_height,
                border_mode=cv2.BORDER_REPLICATE, keep_aspect_ratio=keep_aspect_ratio)
            diff_ref = np.abs(ref_roi_tanh_frame.astype(int) - roi_tanh_frame.astype(int)).astype(np.uint8)
    else:
        diff_ref = None
    return roi_tanh_polar_frame, roi_tanh_frame, restored_frame, diff_ref, diff_direct


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--video', '-v', help='Video source')
    parser.add_argument('--width', '-x', help='Width of the warped image (default=256)', type=int, default=256)
    parser.add_argument('--height', '-y', help='Height of the warped image (default=256)', type=int, default=256)
    parser.add_argument('--offset', '-o', help='Angular offset, only used when polar>0', type=float, default=0.0)
    parser.add_argument('--restore', '-r', help='Show restored frames',
                        action='store_true', default=False)
    parser.add_argument('--compare', '-c', help='Compare with reference implementation',
                        action='store_true', default=False)
    parser.add_argument('--compare-direct', '-t', help='Compare with directly warped frames',
                        action='store_true', default=False)
    parser.add_argument('--square', '-s', help='Use square-shaped detection box',
                        action='store_true', default=False)
    parser.add_argument('--keep-aspect-ratio', '-k', help='Keep aspect ratio in tanh-polar or tanh-circular warping',
                        action='store_true', default=False)
    parser.add_argument('--reverse', '-i', help='Perform computation in the reverse direction',
                        action='store_true', default=False)
    parser.add_argument('--device', '-d', help='Device to be used (default=cuda:0)', default='cuda:0')
    parser.add_argument('--benchmark', '-b', help='Enable benchmark mode for CUDNN',
                        action='store_true', default=False)
    args = parser.parse_args()

    # Make the models run a bit faster
    torch.backends.cudnn.benchmark = args.benchmark

    # Create face detector
    detector = RetinaFacePredictor(device=args.device, model=RetinaFacePredictor.get_model('mobilenet0.25'))
    print('RetinaFace detector created using mobilenet0.25 backbone.')

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
                face_boxes = detector(frame, rgb=False)
                if len(face_boxes) > 0:
                    biggest_face_idx = int(np.argmax([(bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
                                                      for bbox in face_boxes]))

                    # Test the warping functions
                    start_time = time.time()
                    roi_tanh_polar_frame, roi_tanh_frame, restored_frame, diff_ref, diff_direct = test_pytorch_impl(
                        args.device, frame, face_boxes[biggest_face_idx], args.width, args.height,
                        args.offset / 180.0 * np.pi, args.restore, args.compare, args.compare_direct,
                        args.square, args.keep_aspect_ratio, args.reverse)
                    elapsed_time = time.time() - start_time
                    print(f'Frame #{frame_number}: Warped and processed in {elapsed_time * 1000.0: .1f} ms.')

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
                    diff_ref = None
                    diff_direct = None
                    print(f'Frame #{frame_number}: No face detected.')

                # Show the result
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
                if args.compare_direct:
                    if diff_direct is None:
                        cv2.destroyWindow('Diff-w-Direct')
                    else:
                        cv2.imshow('Diff-w-Direct', diff_direct)
                if args.compare:
                    if diff_ref is None:
                        cv2.destroyWindow('Diff-w-Ref')
                    else:
                        cv2.imshow('Diff-w-Ref', diff_ref)
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
