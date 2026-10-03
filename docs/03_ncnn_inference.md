# 03 — NCNN Inference on Raspberry Pi 4

This guide runs a YOLO model exported in **NCNN FP32** format on the 
Raspberry Pi 4. NCNN is optimized for ARM CPUs and delivers higher 
detection confidence than INT8 quantization.

**Expected performance on RPi 4 (4GB):**
- FPS: ~6.5 at 320×320 input
- Confidence: 0.80+ on clear objects
- CPU utilization: 80–90%
- Max temperature: 55°C with active cooler

---

## What is NCNN?

NCNN is a high-performance neural network inference framework built by 
**Tencent**, optimized specifically for mobile and embedded ARM CPUs. 

Key properties:
- **Format:** `.param` (network structure, text) + `.bin` (weights, binary)
- **Precision:** FP32 by default (also supports FP16 and INT8)
- **Loads via Ultralytics:** You do not need to install `ncnn` separately — 
  Ultralytics has a bundled NCNN backend

**Why use NCNN?**
- Higher accuracy than INT8 quantization
- Best for scenes where missed detections matter
- Best for slow-moving or stationary objects

**Trade-off:**
- ~2× slower than TFLite INT8 (6.5 FPS vs 13.75 FPS)

---

## Prerequisites

- NCNN model folder on the Pi (`yolo26n_ncnn_model/`) with `.bin`, 
  `.param`, and `metadata.yaml`
- The inference script (`scripts/yolo_detect_ncnn.py` in this repo, 
  or `yolo_detect_updated.py` in your local setup)
- A test video or image, or a connected Pi camera
- Activated venv (`source ~/yolo/venv/bin/activate`)

Verify the model folder:

`bash
ls ~/yolo/yolo26n_ncnn_model/

metadata.yaml
model.ncnn.bin
model.ncnn.param
model_ncnn.py  

## Step 1 — Review the Inference Script

The script auto-detects the model format. When you pass it a folder
ending in _ncnn_model, Ultralytics loads its NCNN backend.

Key parts of the script:
from ultralytics import YOLO
model = YOLO(model_path, task='detect')   # auto-detects NCNN format
labels = model.names                       # reads from metadata.yaml

Then per frame:
results = model(frame, conf=min_thresh, verbose=False)

That's it. Ultralytics handles NCNN loading, input preprocessing, and
output decoding internally.

## Step 2 — Run Inference on a Video File

With the venv active:

cd ~/yolo
python yolo_detect_updated.py \
    --model yolo26n_ncnn_model \
    --source path/to/video.mp4 \
    --resolution 320x320
	
	Arguments explained:

Flag			Value				Meaning
--model			yolo26n_ncnn_model	Path to the NCNN model folder
--source		video.mp4			Input source (see below for options)
--resolution	320x320				Display resolution
--thresh		(default 0.5)		Confidence threshold
--record		(flag)				Save output to demo1.avi

Expected output:
Loading yolo26n_ncnn_model for NCNN inference...
Loading yolo26n_ncnn_model for NCNN inference...
qt.qpa.plugin: Could not find the Qt platform plugin "wayland"...
Reached end of the video file. Exiting program.
Average pipeline FPS: 6.46

The "Loading ... for NCNN inference..." message is printed twice by
Ultralytics — this is normal, not an error.

The Qt warning about Wayland is harmless. Labels still render using
system fallback fonts.

## Step 3 — Run Inference on the Pi Camera
To use the Pi Camera Module v1.3 as the input source:

python yolo_detect_updated.py \
    --model yolo26n_ncnn_model \
    --source picamera0 \
    --resolution 640x480

The picamera0 source uses picamera2 internally.

Note: The display resolution (640x480) is different from the model
input resolution (320×320). The script resizes internally before
inference.

## Step 4 — Run Inference on a Single Image

python yolo_detect_updated.py \
    --model yolo26n_ncnn_model \
    --source path/to/image.jpg

Press any key to advance to the next image (only matters for folders).

## Step 5 — Run Inference on a Folder of Images

python yolo_detect_updated.py \
    --model yolo26n_ncnn_model \
    --source path/to/image_folder/
	
Press any key to advance between images.

## Step 6 — Record Output

Add --record to save the annotated output as demo1.avi in the
current directory:

python yolo_detect_updated.py \
    --model yolo26n_ncnn_model \
    --source video.mp4 \
    --resolution 320x320 \
    --record

Note: The recorded file is written at 30 FPS, regardless of
actual inference FPS. A 30-second source video with 6.46 FPS inference
produces a demo1.avi that plays at 30 FPS but only shows ~193 frames
of actual detection.

Using Tracking Instead of Detection
The script supports ByteTrack for multi-object tracking. Pass
--task track:

python yolo_detect_updated.py \
    --model yolo26n_ncnn_model \
    --source video.mp4 \
    --task track
	
Optionally add --show-tracks to draw movement trails:

python yolo_detect_updated.py \
    --model yolo26n_ncnn_model \
    --source video.mp4 \
    --task track \
    --show-tracks

Tracking adds CPU overhead. Expect ~1–2 FPS lower than detection-only.

Key	Action
q	Quit
s	Pause inference (press any key to resume)
p	Save current frame as capture.png

   ## What to Expect
## Good detections:
Large objects: cars, buses, people — detected with confidence 0.85+

Well-lit scenes with clear objects

Slow-moving or stationary objects

## Weak detections:
Small objects at distance

Objects in shadow or low light

Fast-moving objects (motion blur)

## Common missed detections:
Objects smaller than 30×30 pixels

Objects partially occluded

Objects outside the training distribution

## Performance Tuning
If FPS is too low:
Reduce input resolution. Re-export the model at imgsz=256 and
run inference at --resolution 256x256. FPS scales roughly with
resolution squared.

Reduce display resolution. --resolution 240x180 reduces drawing
overhead.

Turn off --record. Recording adds write overhead.

Close other processes. RPi 4 is CPU-bound during inference.

If accuracy is too low:
Increase input resolution. Re-export at imgsz=416 or imgsz=640.

Lower confidence threshold. --thresh 0.3 shows more detections
(some are false positives).

Check lighting. NCNN FP32 is still limited by what the camera sees.

	
	

