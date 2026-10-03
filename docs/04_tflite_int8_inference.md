# 04 — TFLite INT8 Inference on Raspberry Pi 4

This guide runs a YOLO model exported in **TFLite INT8** format on the 
Raspberry Pi 4. The quantized model delivers roughly **2× the FPS** of 
NCNN FP32, at the cost of lower confidence on fast-moving objects.

**Expected performance on RPi 4 (4GB):**
- FPS: ~13.75 at 320×320 input
- Confidence: ~0.50 on clear objects
- CPU utilization: 80–90%
- Max temperature: 55°C with active cooler

---

## What is TFLite INT8?

TensorFlow Lite (TFLite) is Google's runtime for on-device inference. In 
2024, Google renamed it to **LiteRT**. The Python package is now 
`ai-edge-litert`.

**INT8 quantization** converts model weights and activations from 32-bit 
floats to 8-bit integers. This:
- **Reduces model size** by ~4×
- **Speeds up inference** by ~2×
- **Loses precision**, especially on small or fast-moving objects

**Why use TFLite INT8?**
- Real-time performance matters most
- Objects are relatively slow-moving
- You need CPU headroom for other tasks
- You want a smaller model file

**Trade-off:**
- Lower confidence than NCNN FP32
- More missed detections on fast motion

---

## How Ultralytics Loads TFLite Models

You do **not** install `tflite-runtime` or `ai-edge-litert` separately. 
Ultralytics detects the model format by file extension (`.tflite`) and 
loads its bundled TFLite backend automatically.

Under the hood, Ultralytics uses **XNNPACK**, an optimized CPU kernel 
library for ARM. You will see this in the startup log:

INFO: Created TensorFlow Lite XNNPACK delegate for CPU.


This is expected and confirms the model is running on the ARM-optimized 
backend.

---

## Prerequisites

- The `.tflite` model file on the Pi (`yolo26n_int8.tflite`)
- The inference script (`yolo_detect_updated.py`)
- A test video, image, or Pi camera
- Activated venv (`source ~/yolo/venv/bin/activate`)

Verify the model:

`bash
ls -la ~/yolo/yolo26n_int8.tflite

Expected: a file around 3 MB. If it's much larger (e.g., 15–20 MB),
quantization did not apply correctly — see File 02 Step 5.

## Step 1 — Run Inference on a Video File 

cd ~/yolo
python yolo_detect_updated.py \
    --model yolo26n_int8.tflite \
    --source path/to/video.mp4 \
    --resolution 320x320
	
Arguments:

Flag			Value					Meaning
--model			yolo26n_int8.tflite		Path to the INT8 model
--source		video.mp4				Input source
--resolution	320x320					Display resolution
--thresh		(default 0.5)			Confidence threshold
--record		(flag)					Save output to demo1.avi

Expected output:

Loading yolo26n_int8.tflite for LiteRT inference...
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Loading yolo26n_int8.tflite for LiteRT inference...
qt.qpa.plugin: Could not find the Qt platform plugin "wayland"...
Reached end of the video file. Exiting program.
Average pipeline FPS: 13.75

## Step 2 — Run Inference on the Pi Camera

python yolo_detect_updated.py \
    --model yolo26n_int8.tflite \
    --source picamera0 \
    --resolution 640x480

Live detection at ~10–13 FPS on the Pi camera. This is where INT8 shines
— the higher FPS makes live scenes feel responsive.

## Step 3 — Run Inference on an Image or Folder
Single image:

python yolo_detect_updated.py \
    --model yolo26n_int8.tflite \
    --source path/to/image.jpg
	
Folder of images:

python yolo_detect_updated.py \
    --model yolo26n_int8.tflite \
    --source path/to/image_folder/
	
## Step 4 — Record Output
Add --record:

python yolo_detect_updated.py \
    --model yolo26n_int8.tflite \
    --source video.mp4 \
    --resolution 320x320 \
    --record

Output saved to demo1.avi at 30 FPS playback. Because actual inference
runs at ~13.75 FPS, the recorded video plays smoothly but skips frames
(only detected frames are drawn; missed frames are not interpolated).

## Strong detections:
Large objects (cars, buses, people) — confidence 0.55–0.75

Static or slow-moving objects

## Weak detections:
Small objects at distance

Fast-moving objects — INT8 misses these more than NCNN

Comparison with NCNN FP32:

Scenario				NCNN FP32			TFLite INT8
Static car				Detected (0.90)		Detected (0.60)
Fast-moving car			Detected (0.75)		Often missed
Walking person			Detected (0.85)		Detected (0.55)
Small distant object	Detected (0.65)		Sometimes missed

The gap is largest on fast motion — motion blur + reduced precision
compound the problem.

## Performance Tuning

#If you need more FPS:

Reduce input resolution. Re-export the model with imgsz=256 and
run at --resolution 256x256.

Turn off --record. Recording adds write overhead.

Use the smaller model. YOLO11n or YOLOv8n may be faster than YOLO26n.

Disable tracking. Detection-only is faster than --task track.

#If accuracy is too low:

Increase input resolution. Re-export at imgsz=416.

Lower the threshold. --thresh 0.3 catches more objects.

Switch to NCNN. If accuracy matters more than speed, use the NCNN
model instead (see File 06).

#When INT8 Is the Right Choice:

Live camera feeds — higher FPS makes the UI feel responsive

Slow-moving scenes — no motion blur to fight

Combined pipelines — you need CPU headroom for tracking, OCR, or
sending results to a dashboard

Memory-constrained — smaller model file leaves more room for other
processes

#When INT8 Is the Wrong Choice:

Fast action (highway traffic, sports, motor racing)

Small object detection at distance

Safety-critical applications where missed detections are unacceptable

Scientific measurement requiring high precision
	
## Comparison Summary

NCNN 				FP32		TFLiteINT8
FPS					6.46		13.75
Confidence			0.80+		0.50
Model size			~9 MB		~3 MB
Fast-motion misses	Fewer		More
CPU usage			80–90%		80–90%
Recommendation	Accuracy-critical	Real-time

Rule of thumb: If you can tolerate some missed detections and want
smooth live feedback, use INT8. If every detection counts, use NCNN.