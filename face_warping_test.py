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
                      polar: int, offset: float, restore: bool, square: bool, nearest: bool,
                      keep_aspect_ratio: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    # Preparation
    frames = torch.from_numpy(frame.astype(np.float32)).to(device).permute(2, 0, 1).unsqueeze(0)
    face_boxes = torch.from_numpy(face_box[:4]).to(device).unsqueeze(0)
    if square:
        face_boxes = make_square_rois(face_boxes)

    # Warping
    if polar > 1:
        warped_frames = roi_tanh_circular_warp(frames, face_boxes, target_width, target_height,
                                               angular_offsets=offset, padding='border',
                                               keep_aspect_ratio=keep_aspect_ratio)
    elif polar > 0:
        warped_frames = roi_tanh_polar_warp(frames, face_boxes, target_width, target_height,
                                            angular_offsets=offset, padding='border',
                                            keep_aspect_ratio=keep_aspect_ratio)
    else:
        warped_frames = roi_tanh_warp(frames, face_boxes, target_width, target_height,
                                      angular_offsets=offset, padding='border')
    warped_frame = warped_frames[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    # Restoration
    interpolation = 'nearest' if nearest else 'bilinear'
    if restore:
        if polar > 1:
            restored_frames = roi_tanh_circular_restore(warped_frames, face_boxes, *frames.size()[:-3:-1],
                                                        angular_offsets=offset, interpolation=interpolation,
                                                        padding='border', keep_aspect_ratio=keep_aspect_ratio)
        elif polar > 0:
            restored_frames = roi_tanh_polar_restore(warped_frames, face_boxes, *frames.size()[:-3:-1],
                                                     angular_offsets=offset, interpolation=interpolation,
                                                     padding='border', keep_aspect_ratio=keep_aspect_ratio)
        else:
            restored_frames = roi_tanh_restore(warped_frames, face_boxes, *frames.size()[:-3:-1],
                                               angular_offsets=offset, interpolation=interpolation,
                                               padding='border')
        restored_frame = restored_frames[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    else:
        restored_frame = None

    return warped_frame, restored_frame


def test_reference_impl(frame: np.ndarray, face_box: np.ndarray, target_width: int, target_height: int,
                        polar: int, offset: float, restore: bool, square: bool, nearest: bool,
                        keep_aspect_ratio: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    # Preparation
    if square:
        face_box = ref.make_square_rois(face_box[:4])

    # Warping
    if polar > 1:
        warped_frame = ref.roi_tanh_circular_warp(frame, face_box, target_width, target_height,
                                                  angular_offset=offset, border_mode=cv2.BORDER_REPLICATE,
                                                  keep_aspect_ratio=keep_aspect_ratio)
    elif polar > 0:
        warped_frame = ref.roi_tanh_polar_warp(frame, face_box, target_width, target_height,
                                               angular_offset=offset, border_mode=cv2.BORDER_REPLICATE,
                                               keep_aspect_ratio=keep_aspect_ratio)
    else:
        warped_frame = ref.roi_tanh_warp(frame, face_box, target_width, target_height,
                                         angular_offset=offset, border_mode=cv2.BORDER_REPLICATE)

    # Restoration
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    if restore:
        if polar > 1:
            restored_frame = ref.roi_tanh_circular_restore(warped_frame, face_box, *frame.shape[1::-1],
                                                           angular_offset=offset, interpolation=interpolation,
                                                           border_mode=cv2.BORDER_REPLICATE,
                                                           keep_aspect_ratio=keep_aspect_ratio)
        elif polar > 0:
            restored_frame = ref.roi_tanh_polar_restore(warped_frame, face_box, *frame.shape[1::-1],
                                                        angular_offset=offset, interpolation=interpolation,
                                                        border_mode=cv2.BORDER_REPLICATE,
                                                        keep_aspect_ratio=keep_aspect_ratio)
        else:
            restored_frame = ref.roi_tanh_restore(warped_frame, face_box, *frame.shape[1::-1],
                                                  angular_offset=offset, interpolation=interpolation,
                                                  border_mode=cv2.BORDER_REPLICATE)
    else:
        restored_frame = None

    return warped_frame, restored_frame


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--video', '-v', help='Video source')
    parser.add_argument('--width', '-x', help='Width of the warped image (default=256)', type=int, default=256)
    parser.add_argument('--height', '-y', help='Height of the warped image (default=256)', type=int, default=256)
    parser.add_argument('--polar', '-p', help='Use polar coordinates', type=int, default=0)
    parser.add_argument('--offset', '-o', help='Angular offset, only used when polar>0', type=float, default=0.0)
    parser.add_argument('--restore', '-r', help='Show restored frames',
                        action='store_true', default=False)
    parser.add_argument('--compare', '-c', help='Compare with reference implementation',
                        action='store_true', default=False)
    parser.add_argument('--square', '-s', help='Use square-shaped detection box',
                        action='store_true', default=False)
    parser.add_argument('--nearest', '-n', help='Use nearest-neighbour interpolation during restoration',
                        action='store_true', default=False)
    parser.add_argument('--keep-aspect-ratio', '-k', help='Keep aspect ratio in tanh-polar or tanh-circular warping',
                        action='store_true', default=False)
    parser.add_argument('--device', '-d', help='Device to be used by PyTorch (default=cuda:0)', default='cuda:0')
    parser.add_argument('--benchmark', '-b', help='Enable benchmark mode for CUDNN',
                        action='store_true', default=False)
    args = parser.parse_args()

    # Make the models run a bit faster
    torch.backends.cudnn.benchmark = args.benchmark

    # Create object detector
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

                    start_time = time.time()
                    warped_frame, restored_frame = test_pytorch_impl(
                        args.device, frame, face_boxes[biggest_face_idx], args.width, args.height, args.polar,
                        args.offset / 180.0 * np.pi, args.restore, args.square, args.nearest, args.keep_aspect_ratio)
                    if args.compare:
                        ref_warped_frame, ref_restored_frame = test_reference_impl(
                            frame, face_boxes[biggest_face_idx], args.width, args.height, args.polar,
                            args.offset / 180.0 * np.pi, args.restore, args.square, args.nearest,
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
                    warped_frame = None
                    restored_frame = None
                    diff_warped_frame = None
                    diff_restored_frame = None
                    print(f'Frame #{frame_number}: No face detected.')

                # Show the result
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
