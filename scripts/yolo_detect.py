#!/usr/bin/env python3
"""
YOLO Object Detection Script for Raspberry Pi 4

Supports both NCNN (FP32) and TFLite (INT8) model formats via Ultralytics.
The model format is auto-detected from the file extension (`.tflite`) or
folder name (`*_ncnn_model`).

Usage examples:
    # NCNN model on a video
    python yolo_detect.py --model yolo26n_ncnn_model --source video.mp4 --resolution 320x320

    # TFLite INT8 model on the Pi camera
    python yolo_detect.py --model yolo26n_int8.tflite --source picamera0 --resolution 640x480

    # With recording
    python yolo_detect.py --model yolo26n_int8.tflite --source video.mp4 --resolution 320x320 --record

    # With ByteTrack tracking + trails
    python yolo_detect.py --model yolo26n_int8.tflite --source video.mp4 --task track --show-tracks

Controls (while window is open):
    q — Quit
    s — Pause (press any key to resume)
    p — Save current frame as capture.png
"""

import os
import sys
import argparse
import glob
import time
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


# =============================================================================
# Constants
# =============================================================================

IMG_EXT_LIST = ['.jpg', '.jpeg', '.png', '.bmp']
VID_EXT_LIST = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

# Tableau 10 color scheme, cycled by class ID
BBOX_COLORS = [
    (164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133),
    (88, 159, 106), (96, 202, 231), (159, 124, 168), (169, 162, 241),
    (98, 118, 150), (172, 176, 184),
]


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO object detection on Raspberry Pi 4"
    )
    parser.add_argument(
        '--model',
        help='Path to YOLO model file (.tflite) or folder (*_ncnn_model)',
        required=True,
    )
    parser.add_argument(
        '--source',
        help='Image source: image file, image folder, video file, usb0, or picamera0',
        required=True,
    )
    parser.add_argument(
        '--thresh',
        help='Minimum confidence threshold for displaying detected objects',
        default=0.5,
    )
    parser.add_argument(
        '--resolution',
        help='Resolution in WxH to display inference results at (e.g., "640x480")',
        default=None,
    )
    parser.add_argument(
        '--record',
        help='Record results and save as "demo1.avi". Must specify --resolution.',
        action='store_true',
    )
    parser.add_argument(
        '--task',
        help='Task: "detect" or "track"',
        default='detect',
        choices=['detect', 'track'],
    )
    parser.add_argument(
        '--show-tracks',
        help='Draw tracking trails (only for task=track)',
        action='store_true',
    )
    return parser.parse_args()


# =============================================================================
# Helpers
# =============================================================================

def classify_source(source):
    """Return source type: 'folder', 'image', 'video', 'usb', or 'picamera'."""
    if os.path.isdir(source):
        return 'folder'

    if os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in IMG_EXT_LIST:
            return 'image'
        if ext in VID_EXT_LIST:
            return 'video'
        print(f'File extension {ext} is not supported.')
        sys.exit(1)

    if 'usb' in source:
        return 'usb'
    if 'picamera' in source:
        return 'picamera'

    print(f'Input {source} is invalid. Please try again.')
    sys.exit(1)


def list_image_folder(folder):
    """Return list of image files in a folder."""
    files = glob.glob(folder + '/*')
    return [f for f in files if os.path.splitext(f)[1].lower() in IMG_EXT_LIST]


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # ---- Parse user inputs ----
    model_path = args.model
    img_source = args.source
    min_thresh = float(args.thresh)
    user_res = args.resolution
    record = args.record
    task = args.task
    show_tracks = args.show_tracks

    # ---- Validate model path ----
    if not os.path.exists(model_path):
        print('ERROR: Model path is invalid or model was not found.')
        sys.exit(1)

    # ---- Load model ----
    model = YOLO(model_path, task='detect')
    labels = model.names

    # ---- Determine source type ----
    source_type = classify_source(img_source)

    # Extract indices for usb/picamera sources
    usb_idx = None
    picam_idx = None
    if source_type == 'usb':
        usb_idx = int(img_source[3:])
    elif source_type == 'picamera':
        picam_idx = int(img_source[8:])  # extracted for clarity; picamera2 opens camera 0

    # ---- Parse display resolution ----
    resize = False
    resW = resH = None
    if user_res:
        resize = True
        resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

    # ---- Recording setup ----
    recorder = None
    if record:
        if source_type not in ('video', 'usb', 'picamera'):
            print('Recording only works for video, camera, and picamera sources. Please try again.')
            sys.exit(1)
        if not user_res:
            print('Please specify resolution to record video at.')
            sys.exit(1)

        record_name = 'demo1.avi'
        record_fps = 30
        recorder = cv2.VideoWriter(
            record_name,
            cv2.VideoWriter_fourcc(*'MJPG'),
            record_fps,
            (resW, resH),
        )

    # ---- Initialize source ----
    cap = None
    imgs_list = []
    img_count = 0

    if source_type == 'image':
        imgs_list = [img_source]

    elif source_type == 'folder':
        imgs_list = list_image_folder(img_source)
        if not imgs_list:
            print(f'No supported images found in {img_source}')
            sys.exit(1)

    elif source_type == 'video' or source_type == 'usb':
        cap_arg = img_source if source_type == 'video' else usb_idx
        cap = cv2.VideoCapture(cap_arg)
        if user_res:
            cap.set(3, resW)
            cap.set(4, resH)

    elif source_type == 'picamera':
        from picamera2 import Picamera2
        # Bug fix: default to 640x480 if no --resolution was provided
        cam_w, cam_h = (resW, resH) if user_res else (640, 480)
        cap = Picamera2()
        cap.configure(cap.create_video_configuration(
            main={"format": 'XRGB8888', "size": (cam_w, cam_h)}
        ))
        cap.start()

    # ---- Runtime state ----
    avg_frame_rate = 0.0
    frame_rate_buffer = []
    fps_avg_len = 200
    track_history = defaultdict(list)

    # =========================================================================
    # Inference loop
    # =========================================================================

    while True:
        t_start = time.perf_counter()

        # ---- Read frame ----
        if source_type in ('image', 'folder'):
            if img_count >= len(imgs_list):
                print('All images have been processed. Exiting program.')
                break
            frame = cv2.imread(imgs_list[img_count])
            img_count += 1

        elif source_type == 'video':
            ret, frame = cap.read()
            if not ret:
                print('Reached end of the video file. Exiting program.')
                break

        elif source_type == 'usb':
            ret, frame = cap.read()
            if (frame is None) or (not ret):
                print('Unable to read frames from the camera. Exiting program.')
                break

        elif source_type == 'picamera':
            frame_bgra = cap.capture_array()
            frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
            if frame is None:
                print('Unable to read frames from the Picamera. Exiting program.')
                break

        # ---- Resize for display ----
        if resize:
            frame = cv2.resize(frame, (resW, resH))

        # ---- Inference ----
        if task == 'track':
            results = model.track(
                frame, conf=min_thresh, persist=True,
                tracker="bytetrack.yaml", verbose=False,
            )
        else:
            results = model(frame, conf=min_thresh, verbose=False)

        detections = results[0].boxes
        object_count = 0

        # ---- Draw detections ----
        for i in range(len(detections)):

            # Bounding box coordinates
            xyxy = detections[i].xyxy.cpu().numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            # Class and confidence
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = detections[i].conf.item()

            # Draw only if above threshold (matches original strict >)
            if conf > min_thresh:

                color = BBOX_COLORS[classidx % 10]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                label = f'{classname}: {int(conf * 100)}%'
                labelSize, baseLine = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(
                    frame,
                    (xmin, label_ymin - labelSize[1] - 10),
                    (xmin + labelSize[0], label_ymin + baseLine - 10),
                    color, cv2.FILLED,
                )
                cv2.putText(
                    frame, label, (xmin, label_ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                )

                object_count += 1

                # ---- Tracking trails ----
                if task == 'track' and show_tracks:
                    if detections[i].id is not None:
                        track_id = int(detections[i].id.item())
                        cx = (xmin + xmax) / 2
                        cy = (ymin + ymax) / 2
                        track = track_history[track_id]
                        track.append((float(cx), float(cy)))
                        if len(track) > 30:
                            track.pop(0)
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [points], isClosed=False, color=color, thickness=3)

        # ---- FPS overlay ----
        if source_type in ('video', 'usb', 'picamera'):
            cv2.putText(
                frame, f'FPS: {avg_frame_rate:0.2f}', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2,
            )

        # ---- Object count overlay ----
        cv2.putText(
            frame, f'Number of objects: {object_count}', (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2,
        )

        # ---- Show + record ----
        cv2.imshow('YOLO detection results', frame)
        if record:
            recorder.write(frame)

        # ---- Key handling ----
        if source_type in ('image', 'folder'):
            key = cv2.waitKey()
        else:
            key = cv2.waitKey(5)

        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            cv2.waitKey()
        elif key == ord('p') or key == ord('P'):
            cv2.imwrite('capture.png', frame)

        # ---- Frame rate calculation ----
        t_stop = time.perf_counter()
        frame_rate_calc = float(1 / (t_stop - t_start))

        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
            frame_rate_buffer.append(frame_rate_calc)
        else:
            frame_rate_buffer.append(frame_rate_calc)

        avg_frame_rate = np.mean(frame_rate_buffer)

    # =========================================================================
    # Cleanup
    # =========================================================================
    print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
    if source_type in ('video', 'usb'):
        cap.release()
    elif source_type == 'picamera':
        cap.stop()
    if record:
        recorder.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()