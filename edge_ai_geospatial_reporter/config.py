"""
config.py - Central configuration for the Edge-AI Geospatial Infrastructure & Asset Reporter.

Handles filesystem paths, CUDA/device configuration, and all tunable
thresholds used across inference, database, reporting, and the dashboard.
"""

import os
from pathlib import Path
import torch

# --------------------------------------------------------------------------
# Filesystem paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "feed_images"        # simulated video feed / drone image drop folder
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

for _dir in (DATA_DIR, IMAGES_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = DATA_DIR / "asset_reports.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# YOLO model weights. Defaults to the small pretrained COCO checkpoint so the
# project runs out of the box; swap for a custom-trained infrastructure model
# (e.g. 'infra_yolov8.pt') by dropping it into MODELS_DIR and updating this path.
MODEL_WEIGHTS_PATH = MODELS_DIR / "yolov8n.pt"

# --------------------------------------------------------------------------
# Device / CUDA configuration - tuned for a 6GB VRAM budget (RTX 3050)
# --------------------------------------------------------------------------
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda:0" if CUDA_AVAILABLE else "cpu"

# Half precision (fp16) is used on GPU to roughly halve activation/weight
# memory footprint, which matters a lot on a 6GB card.
USE_FP16 = CUDA_AVAILABLE

# Cap VRAM usage so YOLO/PyTorch don't try to grab everything available.
# Expressed as a fraction of total device memory (0.0 - 1.0).
MAX_VRAM_FRACTION = 0.85

# Batch size for inference. Kept conservative for 6GB VRAM.
GPU_BATCH_SIZE = 4
CPU_BATCH_SIZE = 1

# Number of worker threads used for the CPU fallback pipeline
# (Ryzen 7 typically has 8 cores / 16 threads; leave headroom for the OS/UI).
CPU_WORKER_THREADS = max(1, (os.cpu_count() or 8) - 2)

# Inference image size (square, pixels). 640 is the YOLOv8 default and a good
# accuracy/VRAM tradeoff for 6GB cards.
INFERENCE_IMG_SIZE = 640

if CUDA_AVAILABLE:
    try:
        torch.cuda.set_per_process_memory_fraction(MAX_VRAM_FRACTION, device=0)
    except Exception:
        # Some driver/toolkit combinations don't support this call; safe to ignore.
        pass

# --------------------------------------------------------------------------
# Anomaly / asset class configuration
# --------------------------------------------------------------------------
# Maps COCO-pretrained class names to the infrastructure-anomaly vocabulary
# used throughout the dashboard. Replace with your fine-tuned model's real
# class list when you swap in a custom checkpoint.
ANOMALY_CLASSES = {
    "pothole": "Pothole / Road Surface Damage",
    "crack": "Structural Crack",
    "encroachment": "Illegal Encroachment",
    "debris": "Debris / Obstruction",
    "flooding": "Waterlogging / Flooding",
    "vegetation_overgrowth": "Vegetation Overgrowth",
    "damaged_pole": "Damaged Utility Pole",
    "exposed_wiring": "Exposed Wiring / Cable Fault",
}

# Global default confidence threshold (0.0 - 1.0)
DEFAULT_CONFIDENCE_THRESHOLD = 0.45

# Per-class override thresholds. Any class not listed falls back to
# DEFAULT_CONFIDENCE_THRESHOLD. Safety-critical classes get a lower bar
# (catch more, tolerate more false positives) while cosmetic ones get a
# higher bar.
CLASS_CONFIDENCE_THRESHOLDS = {
    "exposed_wiring": 0.30,
    "flooding": 0.35,
    "damaged_pole": 0.35,
    "pothole": 0.45,
    "crack": 0.50,
    "encroachment": 0.50,
    "debris": 0.45,
    "vegetation_overgrowth": 0.55,
}

# Severity ranking used for sorting/highlighting in the dashboard and report.
SEVERITY_ORDER = {
    "exposed_wiring": 1,
    "flooding": 2,
    "damaged_pole": 3,
    "encroachment": 4,
    "pothole": 5,
    "crack": 6,
    "debris": 7,
    "vegetation_overgrowth": 8,
}

# --------------------------------------------------------------------------
# Simulated GPS flight-path configuration
# --------------------------------------------------------------------------
# Used by inference.py to stamp each processed frame with a plausible
# latitude/longitude when no EXIF GPS data is embedded in the source images.
# Defaults to a bounding box around Chennai, India.
SIM_GPS_CENTER_LAT = 13.0827
SIM_GPS_CENTER_LON = 80.2707
SIM_GPS_STEP_DEGREES = 0.0009   # ~100m per simulated frame step
SIM_GPS_JITTER_DEGREES = 0.0002

# --------------------------------------------------------------------------
# Reporting / email configuration
# --------------------------------------------------------------------------
REPORT_TITLE = "Edge-AI Geospatial Infrastructure & Asset Report"
REPORT_TOP_N_ANOMALIES = 10

# SMTP settings are pulled from environment variables so credentials never
# live in source control. All are optional; email sending is skipped
# gracefully if SMTP_HOST / SMTP_USERNAME are not configured.
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_USE_TLS = os.environ.get("SMTP_USE_TLS", "true").lower() == "true"
REPORT_EMAIL_FROM = os.environ.get("REPORT_EMAIL_FROM", SMTP_USERNAME)
REPORT_EMAIL_TO = [
    addr.strip()
    for addr in os.environ.get("REPORT_EMAIL_TO", "").split(",")
    if addr.strip()
]

# --------------------------------------------------------------------------
# Streamlit dashboard configuration
# --------------------------------------------------------------------------
APP_TITLE = "Edge-AI Geospatial Infrastructure & Asset Reporter"
APP_ICON = "\U0001F6F0\uFE0F"
MAP_DEFAULT_ZOOM = 13
DATAFRAME_REFRESH_SECONDS = 5
