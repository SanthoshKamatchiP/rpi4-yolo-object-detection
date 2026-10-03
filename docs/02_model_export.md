# 02 — Model Export Using Google Colab

This guide exports a YOLO model (`.pt`) into two deployment-ready formats:

1. **NCNN** (FP32) — higher accuracy, lower FPS
2. **TFLite INT8** (quantized) — higher FPS, lower accuracy

Both exports are done in **Google Colab** (free T4 GPU) to keep the Pi 4 
free and use much faster hardware for the conversion.

**Time required:** ~10 minutes for both exports

---

## Why Export on Colab, Not on the Pi?

Exporting a YOLO model requires:
- ~2 GB of temporary disk space
- Significant CPU time (~2–5 minutes on the Pi 4, but thermal-throttled)
- Multiple Python dependencies that are hard to install on ARM64

Google Colab provides:
- A free T4 GPU for any ops that benefit from it
- Fast NVMe storage
- Pre-installed PyTorch and CUDA
- No thermal limits

**Result:** Export on Colab, download the output, run inference on the Pi.

---

## Prerequisites

- Google account (for Colab and Drive)
- The YOLO `.pt` model you want to export (either pretrained or your own)
- Basic familiarity with running Colab cells

If you don't have a `.pt` model yet, you can use the pretrained YOLO26n 
from Ultralytics:

yolo26n.pt # pretrained on COCO (80 classes)


## Step 1 — Open a Colab Notebook

1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. Click **File → New Notebook**
3. Click **Runtime → Change runtime type**
4. Set **Hardware accelerator** to **T4 GPU**
5. Click **Save**

Verify the GPU is active:

`python
!nvidia-smi

You should see a "Tesla T4" listing. If not, re-check runtime settings.

## Step 2 — Install Ultralytics
In the first cell:

!pip install ultralytics -q

Note: Colab often has an older ultralytics cached. The -q flag
suppresses verbose output; the install is still happening

## Step 3 — Get the Base Model
Pretrained YOLO26n (COCO, 80 classes)

!yolo export model=yolo26n.pt

## Step 4 — Export to NCNN (FP32)

!yolo export model=yolo26n.pt format=ncnn imgsz=320

Arguments explained:

Argument	Value	     Why
model	    yolo26n.pt	 Path to the .pt file
format	    ncnn	     Export to NCNN format
imgsz	    320	         Input resolution. Match to your inference target

What you get:

A folder yolo26n_ncnn_model/ containing:

yolo26n_ncnn_model/
├── metadata.yaml       # class names, input size, etc.
├── model.ncnn.bin      # weights (binary)
├── model.ncnn.param    # network structure (text)
└── model_ncnn.py       # example inference script

Optional — also export at 416 or 640 if you want to test higher
resolutions:
!yolo export model=yolo26n.pt format=ncnn imgsz=416
!yolo export model=yolo26n.pt format=ncnn imgsz=640

## Step 5 — Export to TFLite INT8

!yolo export model=yolo26n.pt format=tflite int8=True imgsz=320 data=coco128.yaml

Argument	Value			Why
model		yolo26n.pt		Path to the .pt file
format		tflite			Export to TFLite format
int8		True			Enable INT8 quantization
imgsz		320				Input resolution
data		coco128.yaml	Calibration dataset for quantization

Why data=coco128.yaml is critical:

INT8 quantization requires a calibration dataset — a set of sample
images the exporter runs through the model to determine the dynamic range
of each activation. Without this, Ultralytics silently falls back to
float32, and you don't get the speed benefit.

coco128.yaml is a small 128-image COCO subset that Ultralytics
downloads automatically. It's used only for calibration, not for
retraining.

For custom models: Use your own dataset yaml (e.g., data.yaml from
your training). But any reasonable calibration set works — coco128 is
fine for most cases.

Alternative syntax: In some Ultralytics versions, the flag is
quantize=8 instead of int8=True. Both work. If one fails, try the
other.

What you get:

yolo26n_saved_model/
├── yolo26n_full_integer_quant.tflite     ← INT8 model (use this)
├── yolo26n_float32.tflite                ← float32 (backup)
└── yolo26n_integer_quant.tflite          ← hybrid (unused)

## Step 7 — Zip and Download
For NCNN:
!zip -r yolo26n_ncnn_model.zip yolo26n_ncnn_model/

For INT8:
Just the .tflite file is needed.

from google.colab import files
files.download("yolo26n_ncnn_model.zip")
files.download("yolo26n_int8.tflite")

## Step 8 — Transfer to Raspberry Pi
Use WinSCP or scp to transfer both to your Pi:

# From your PC
scp yolo26n_ncnn_model.zip skrpi@<pi-ip>:~/yolo/
scp yolo26n_int8.tflite    skrpi@<pi-ip>:~/yolo/

On the Pi, unzip the NCNN model:
cd ~/yolo
unzip yolo26n_ncnn_model.zip

You now have:

~/yolo/
├── yolo26n_ncnn_model/       (folder with .bin, .param, metadata.yaml)
└── yolo26n_int8.tflite

Applying This to Other Models
The exact same commands work for any YOLO .pt model. Just swap the
model filename:

Model		Command
YOLO11n		!yolo export model=yolo11n.pt format=ncnn imgsz=320
YOLOv8n		!yolo export model=yolov8n.pt format=ncnn imgsz=320
YOLO26n		!yolo export model=yolo26n.pt format=ncnn imgsz=320
Same for TFLite INT8 — just change the model= argument

TFLite export produces a huge file (>20 MB) with int8=True
The quantization fell back to float32. Check that you included
data=coco128.yaml in the export command. Without calibration data,
Ultralytics cannot quantize.

NCNN export produces model with wrong input size
Ensure imgsz=320 matches what you'll use at inference. If you exported
at 320 but run inference at 416, the model will fail with a shape mismatch.

Exported model accuracy is terrible
INT8 quantization can degrade accuracy by 30–40% on small objects. For
accuracy-critical use, use NCNN FP32 instead.
