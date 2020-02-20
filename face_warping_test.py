import os
import cv2
import torch
import numpy as np
from argparse import ArgumentParser
from saic_vision.object_detector import S3FMobileV2Detector
from roi_tanh_warping import *


def main():
    parser = ArgumentParser()
    parser.add_argument('-v', '--video', help='video source')
    parser.add_argument('-x', '--width', help='face width', type=int, default=256)
    parser.add_argument('-y', '--height', help='face height', type=int, default=256)
    parser.add_argument('-p', '--polar', help='use polar coordinates', type=int, default=0)
    parser.add_argument('-r', '--restore', help='show restored frames', action='store_true')
    parser.add_argument('-o', '--offset', help='angular offset, only used when polar=1', type=float, default=0.0)
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
                    if args.polar > 1:
                        warped_frame = roi_tanh_circular_warp(frame, face_boxes[biggest_face_idx],
                                                              (args.width, args.height),
                                                              border_mode=cv2.BORDER_REPLICATE)
                    elif args.polar > 0:
                        warped_frame = roi_tanh_polor_warp(frame, face_boxes[biggest_face_idx],
                                                           (args.width, args.height),
                                                           angular_offset=args.offset * np.pi,
                                                           border_mode=cv2.BORDER_REPLICATE)
                    else:
                        warped_frame = roi_tanh_warp(frame, face_boxes[biggest_face_idx],
                                                     (args.width, args.height), border_mode=cv2.BORDER_REPLICATE)

                    # Restoration
                    if args.restore:
                        if args.polar > 1:
                            restored_frame = roi_tanh_circular_restore(warped_frame, face_boxes[biggest_face_idx],
                                                                       frame.shape[1::-1],
                                                                       border_mode=cv2.BORDER_REPLICATE)
                        elif args.polar > 0:
                            restored_frame = roi_tanh_polar_restore(warped_frame, face_boxes[biggest_face_idx],
                                                                    frame.shape[1::-1],
                                                                    angular_offset=args.offset * np.pi,
                                                                    border_mode=cv2.BORDER_REPLICATE)
                        else:
                            restored_frame = roi_tanh_restore(warped_frame, face_boxes[biggest_face_idx],
                                                              frame.shape[1::-1], border_mode=cv2.BORDER_REPLICATE)
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
