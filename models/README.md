# models/

This folder intentionally does **not** contain any model files in the 
repository.

---

## Why No Model Files?

YOLO model files are:

- **Large** — even nano variants are 3–9 MB each. Larger variants 
  (small, medium, large) are 20–150 MB.
- **Easily regenerated** — anyone can create them from a `.pt` file in 
  under 5 minutes using Google Colab.
- **Version-dependent** — an exported model is tied to the exact 
  Ultralytics version, Python version, and export flags used to create it.

Keeping models out of the repo keeps it:
- Under GitHub's 100 MB per-file limit
- Fast to clone
- Free of stale binaries

---

## How to Generate the Models

Follow the guide:

**[docs/02_model_export.md](../docs/02_model_export.md)**

That doc walks through:
1. Opening Google Colab with a T4 GPU
2. Installing Ultralytics
3. Exporting to **NCNN FP32** (for higher accuracy)
4. Exporting to **TFLite INT8** (for higher FPS)
5. Downloading the files
6. Transferring them to the Raspberry Pi

---

## Expected Files After Export

Once you complete `docs/02_model_export.md`, you should have these files 
on your Raspberry Pi:

~/yolo/
├── yolo26n_ncnn_model/ # NCNN FP32 (accuracy-first)
│ ├── metadata.yaml
│ ├── model.ncnn.bin
│ ├── model.ncnn.param
│ └── model_ncnn.py
└── yolo26n_int8.tflite # TFLite INT8 (speed-first)


Approximate sizes:

| File              | Size  |
|---                |---    |
| NCNN FP32 folder  | ~9 MB |
| TFLite INT8 model | ~3 MB |

---

## Using Your Own Trained Model

If you've trained a custom YOLO model (e.g., on your own dataset), 
substitute your `.pt` file for `yolo26n.pt` in the export commands. The 
resulting NCNN/TFLite files will contain your custom classes.

Example:

`python
# In Colab
!yolo export model=my_custom_model.pt format=ncnn imgsz=320
!yolo export model=my_custom_model.pt format=tflite int8=True imgsz=320 data=coco128.yaml