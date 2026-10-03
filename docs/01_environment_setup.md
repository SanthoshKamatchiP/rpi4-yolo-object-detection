# 01 — Environment Setup on Raspberry Pi 4

This guide sets up a Python virtual environment on a Raspberry Pi 4 (4GB) 
running Raspberry Pi OS 64-bit, with all packages needed to run YOLO 
inference via **Ultralytics**.

Ultralytics has built-in backends for both **NCNN** and **TFLite**, so you 
do **not** need to install `ncnn`, `tflite-runtime`, or any additional 
runtime libraries. Ultralytics auto-detects the model format and loads the 
correct backend internally.

**Time required:** ~20 minutes (plus package download time)

---

## Prerequisites

| Requirement | Version / Notes |
|---|---|
| Board | Raspberry Pi 4 Model B (4GB minimum) |
| OS | Raspberry Pi OS 64-bit (Bookworm or Trixie) |
| Python | 3.11 or newer (3.13 tested) |
| Cooling | Active cooler recommended — sustained inference runs hot |
| Storage | 16 GB microSD or USB boot drive minimum |
| Network | Internet access for `apt` and `pip` |

Verify your setup:

#bash
uname -m          # should print: aarch64
python3 --version # should print: Python 3.11+ (3.13 tested)
cat /etc/os-release | grep PRETTY_NAME

If uname -m shows armv7l (32-bit), you are on the wrong OS. Reinstall
Raspberry Pi OS 64-bit before continuing.

 # Step 1 — Update the System

sudo apt update
sudo apt upgrade -y

# Step 2 — Install System Dependencies

sudo apt install -y \
    python3-venv \
    python3-pip \
    python3-opencv \
    python3-picamera2 \
    libcap-dev \
    libcamera-dev \
    ffmpeg
	
#	Notes on specific packages:

python3-picamera2 — installed system-wide. The venv accesses it via
--system-site-packages (Step 3). Required if you want to use the
Pi camera as the input source.

python3-opencv — system OpenCV. The venv sees this via
--system-site-packages.

libcap-dev, libcamera-dev — required for picamera2 and libcamera
headers.

ffmpeg — for video resizing (optional, see scripts/resize_video.sh).


 # Step 3 — Create the Virtual Environment
Create a venv with --system-site-packages so it can access the
system-installed picamera2 and opencv:

mkdir -p ~/yolo
cd ~/yolo
python3 -m venv --system-site-packages venv
source venv/bin/activate

Your prompt should now show (venv) at the start.

Why --system-site-packages?

Without it, the venv cannot see system-installed packages like picamera2
and cv2. Installing them via pip inside the venv is possible but
error-prone on Raspberry Pi OS (PEP 668 restrictions, missing system
headers, etc.). Using --system-site-packages is the standard approach.

Alternative: If you prefer a fully isolated venv, use
python3 -m venv venv (without the flag) and install picamera2 and
opencv-python via pip inside the venv. Both approaches work.

 # Important: The venv must be activated every time you open a new terminal:

cd ~/yolo
source venv/bin/activate
To deactivate: deactivate

# Step 4 — Install Python Packages

With the venv active:
pip install --upgrade pip
pip install ultralytics
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu

That's it. Only two pip install commands are needed.

Why only these two packages?

Ultralytics pulls in everything it needs as dependencies, including:

Its own inference backends for NCNN, TFLite, ONNX, and others

NumPy, Pillow, PyYAML, and other utilities

A bundled set of runtime loaders that automatically detect and load
the correct model format based on file extension

You do not need to install:

❌ ncnn — Ultralytics loads NCNN models via its own bundled backend

❌ tflite-runtime or ai-edge-litert — Ultralytics bundles the
TFLite runtime (LiteRT)

❌ opencv-python — system python3-opencv is visible via the venv

Why CPU-only PyTorch?

The --index-url https://download.pytorch.org/whl/cpu flag installs the
CPU-only build of PyTorch, which is much smaller (~200 MB vs ~800 MB) and
sufficient for inference on the Pi 4. The Pi has no NVIDIA GPU, so the
CUDA build would be dead weight.

Why pin torch==2.6.0 and torchvision==0.21.0?

These are the versions tested and confirmed working on Raspberry Pi OS
64-bit with Python 3.13. Later versions may work but are untested here.

# Step 5 — Verify the Installation

python3 -c "import torch; print('torch', torch.__version__)"
python3 -c "import cv2; print('opencv', cv2.__version__)"
python3 -c "from picamera2 import Picamera2; print('picamera2 OK')"
python3 -c "from ultralytics import YOLO; print('ultralytics OK')"

Expected output 

torch 2.6.0+cpu
opencv 4.x.x
picamera2 OK
ultralytics OK

# Step 6 — Test the Camera

rpicam-hello -t 3000