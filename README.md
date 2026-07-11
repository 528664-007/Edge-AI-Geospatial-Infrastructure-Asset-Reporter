<div align="center">

# 🛰️ Edge-AI Geospatial Infrastructure & Asset Reporter

**Local, edge-deployable computer vision for infrastructure and asset monitoring —
detect, geotag, log, visualize, and report anomalies without ever touching the cloud.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-D71F00?style=flat-square)](https://www.sqlalchemy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=flat-square)]()

[Overview](#-overview) •
[Features](#-key-features) •
[Architecture](#-architecture) •
[Quick Start](#-quick-start) •
[Usage](#-usage-guide) •
[Configuration](#-configuration) •
[Troubleshooting](#-troubleshooting)

</div>

---

## 📖 Overview

The **Edge-AI Geospatial Infrastructure & Asset Reporter** is a fully local,
production-ready pipeline for detecting infrastructure defects and asset
anomalies — potholes, structural cracks, illegal encroachments, exposed
wiring, waterlogging, damaged utility poles, debris, and vegetation
overgrowth — from image feeds, geotagging each detection, and turning the
results into a live operational dashboard and automated PDF reports.

It's built to run entirely on commodity edge hardware: a single **NVIDIA
RTX 3050 (6GB VRAM)** for GPU-accelerated inference, with a graceful,
fully-functional **multi-threaded CPU fallback** for machines without a
discrete GPU. No cloud API calls, no external inference service, no
recurring cost — the entire stack, from detection to reporting, runs on
your machine.

> **Why this exists:** Municipal and facilities teams doing manual visual
> inspections (roads, utility corridors, rights-of-way, campus assets)
> generate large volumes of photo/video evidence but have no fast way to
> triage it. This project turns a folder of inspection images into a
> geotagged, queryable, reportable anomaly log — in minutes, on a laptop.

---

## ✨ Key Features

| | |
|---|---|
| 🎯 **Multi-class anomaly detection** | YOLOv8-based detector covering 8 infrastructure/asset anomaly classes out of the box, remappable to any fine-tuned checkpoint |
| ⚡ **6GB-VRAM optimized inference** | `torch.cuda.amp.autocast()` mixed precision + fp16 weights + capped memory fraction + batch-wise cache clearing |
| 🧵 **Automatic CPU fallback** | Zero-config `ThreadPoolExecutor` pipeline kicks in when no CUDA device is found — same code path, same outputs |
| 🗺️ **Dual map rendering** | PyDeck (`ScatterplotLayer`) for a fast native map, plus an interactive Folium view via `streamlit-folium` |
| 🗄️ **Structured detection ledger** | Every detection persisted to SQLite via SQLAlchemy: timestamp, anomaly type, confidence, lat/lon, bounding box, source image |
| 📊 **Live operational dashboard** | Streamlit UI with per-class threshold sliders, KPI tiles, a live-refreshing table, and a breakdown chart |
| 📄 **One-click PDF reporting** | ReportLab-generated summary report (KPIs, per-type breakdown, recent anomalies table) with optional automatic email delivery |
| 🛰️ **Simulated geolocation** | A flight-path simulator stamps plausible, continuously-advancing GPS coordinates onto frames without EXIF GPS data |
| 🔌 **Zero external dependencies at runtime** | No cloud inference, no paid APIs — everything after `pip install` runs offline |

---

## 🏗️ Architecture

mermaid
flowchart TD
    A["📁 data/feed_images/<br/>simulated image feed"] --> B["🧠 inference.py<br/>YOLOv8 · fp16 autocast (GPU)<br/>or ThreadPoolExecutor (CPU)"]
    B --> C["🗄️ database.py<br/>SQLite via SQLAlchemy"]
    C --> D["📊 app.py<br/>Streamlit dashboard<br/>PyDeck + Folium maps"]
    C --> E["📄 reporter.py<br/>ReportLab PDF + SMTP email"]
    D -.->|"Run Detection Pipeline"| B
    D -.->|"Generate PDF Report"| E


**Data flow, end to end:**
1. Images land in `data/feed_images/` (a stand-in for a live drone/CCTV feed).
2. `inference.py` runs YOLOv8 over each frame, remaps raw classes onto the
   anomaly vocabulary, applies per-class confidence thresholds, and attaches
   a simulated GPS coordinate.
3. Every detection that clears its threshold is bulk-inserted into
   `database.py`'s `DetectionLog` table.
4. `app.py` queries that table on a configurable cache-refresh cadence to
   drive the live table, KPIs, chart, and both maps.
5. `reporter.py` queries the same table to assemble a PDF summary and,
   if SMTP is configured, emails it automatically.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Detection model | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Inference runtime | [PyTorch](https://pytorch.org/) 2.2+ with CUDA AMP |
| Frontend / dashboard | [Streamlit](https://streamlit.io/) |
| Mapping | [PyDeck](https://deckgl.readthedocs.io/) + [Folium](https://python-visualization.github.io/folium/) via `streamlit-folium` |
| Persistence | [SQLAlchemy](https://www.sqlalchemy.org/) ORM over SQLite |
| Reporting | [ReportLab](https://www.reportlab.com/) (Platypus layout engine) |
| Email dispatch | `smtplib` / `email.mime` (stdlib) |
| Data handling | pandas, NumPy, Pillow, OpenCV (headless) |

---

## 💻 Hardware Profile

| Component | Target spec | Behavior |
|---|---|---|
| GPU | NVIDIA RTX 3050, 6GB VRAM | fp16 weights, `autocast()` mixed precision, batch size 4, 85% VRAM cap, cache clear between batches |
| CPU | AMD Ryzen 7 (8c/16t) | `ThreadPoolExecutor` sized to `cpu_count() - 2`; used automatically when no CUDA device is detected |
| RAM | 8GB+ recommended | SQLite + Streamlit + model weights fit comfortably |
| Storage | Minimal | SQLite DB and generated PDFs are lightweight; model checkpoint is the largest artifact (~6MB for `yolov8n.pt`) |

No GPU present? Nothing to configure — `config.py` detects `torch.cuda.is_available() == False` and the entire pipeline transparently switches to the threaded CPU path with identical outputs.

---

## 📂 Project Structure


edge_ai_geospatial_reporter/
├── app.py                 # Streamlit dashboard — sidebar, live table, PyDeck/Folium maps
├── config.py               # Paths, CUDA/device config, class thresholds, SMTP + GPS simulation config
├── database.py              # SQLAlchemy model + CRUD/query functions (SQLite)
├── inference.py             # YOLOv8 pipeline — GPU AMP path + CPU threaded fallback
├── reporter.py               # ReportLab PDF builder + optional SMTP email dispatch
├── requirements.txt
├── data/
│   └── feed_images/        # Drop simulated feed images here (.jpg/.png/.bmp/.webp)
├── models/                  # Drop a fine-tuned YOLOv8 .pt checkpoint here
├── reports/                  # Generated PDF reports land here
└── logs/


---

## 🚀 Quick Start


# 1. Unzip and enter the project
unzip edge_ai_geospatial_reporter.zip
cd edge_ai_geospatial_reporter

# 2. Create an isolated environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. Install PyTorch for your hardware FIRST
#    GPU (RTX 3050, CUDA 12.1 example):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#    CPU-only machines can skip the line above.

# 4. Install the rest of the stack
pip install -r requirements.txt

# 5. Verify device detection
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 6. Add images to process
cp /path/to/your/images/*.jpg data/feed_images/

# 7. Launch the dashboard
streamlit run app.py


Open the URL Streamlit prints (default `http://localhost:8501`), tune thresholds in the sidebar, and click **Run Detection Pipeline**.

---

## ⚙️ Configuration

All tunables live in `config.py`. The most relevant ones:

| Setting | Default | Purpose |
|---|---|---|
| `DEFAULT_CONFIDENCE_THRESHOLD` | `0.45` | Global fallback minimum confidence |
| `CLASS_CONFIDENCE_THRESHOLDS` | per-class dict | Overrides per anomaly type (e.g. `exposed_wiring: 0.30`) |
| `GPU_BATCH_SIZE` | `4` | Images per GPU inference batch — tuned for 6GB VRAM |
| `MAX_VRAM_FRACTION` | `0.85` | Caps total VRAM usage via `torch.cuda.set_per_process_memory_fraction` |
| `CPU_WORKER_THREADS` | `cpu_count() - 2` | Thread pool size for the CPU fallback path |
| `INFERENCE_IMG_SIZE` | `640` | YOLO inference resolution |
| `MODEL_WEIGHTS_PATH` | `models/yolov8n.pt` | Swap in your fine-tuned checkpoint here |
| `SIM_GPS_CENTER_LAT/LON` | Chennai, India | Center point for the simulated flight path |
| `DATAFRAME_REFRESH_SECONDS` | `5` | Dashboard cache/live-refresh cadence |

**Email delivery** (optional) is configured entirely via environment variables, so credentials never touch source control:


export SMTP_HOST=smtp.yourprovider.com
export SMTP_PORT=587
export SMTP_USERNAME=you@yourdomain.com
export SMTP_PASSWORD=your-app-password
export REPORT_EMAIL_TO=ops-team@yourdomain.com,manager@yourdomain.com


If unset, `reporter.py` still builds the PDF and offers it for download — it just skips the email step.

---

## 📘 Usage Guide

### Run detection standalone (no UI)

python inference.py

Prints a summary: `images_processed`, `detections_found`, `device_used`, `elapsed_seconds`.

### Run the dashboard

streamlit run app.py

- **Sidebar** — per-class confidence sliders, queued image count, **Run Detection Pipeline**, **Generate PDF Report**, **Clear All Detections**.
- **Main panel** — KPI tiles, live detection table, anomaly-type bar chart, PyDeck map, expandable Folium map.

### Generate a report programmatically

python reporter.py

Builds a timestamped PDF in `reports/` and attempts email delivery if SMTP is configured.

### Query the database directly

import database
database.get_summary_stats()
database.get_recent_anomalies(limit=10)


---

## 🗄️ Database Schema

`DetectionLog` (`database.py`):

| Column | Type | Notes |
|---|---|---|
| `id` | Integer, PK | Autoincrement |
| `timestamp` | DateTime | Indexed, UTC |
| `anomaly_type` | String(64) | Indexed, keys into `config.ANOMALY_CLASSES` |
| `confidence` | Float | 0.0 – 1.0 |
| `latitude` / `longitude` | Float | Simulated or real GPS |
| `source_image` | String(512) | Path to the originating frame |
| `bbox_x1/y1/x2/y2` | Float | Pixel-space bounding box |
| `notes` | String(256) | Optional free text |

---

## 🔧 VRAM Optimization Details (RTX 3050 / 6GB)

- Model weights cast to fp16 via `model.model.half()` when CUDA is present.
- Every GPU forward pass wrapped in `torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)`.
- Inference batched at `GPU_BATCH_SIZE = 4` — conservative enough to avoid OOM on 6GB while still saturating the GPU.
- `torch.cuda.empty_cache()` called between batches to prevent fragmentation-driven OOMs on long runs.
- `torch.cuda.set_per_process_memory_fraction(0.85)` caps total VRAM draw, leaving headroom for the OS/display.
- CPU fallback uses threads (not processes) since Ultralytics/PyTorch release the GIL during the actual forward pass — no IPC overhead, no pickling cost.

---

## 🩺 Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` | Batch too large for available VRAM | Lower `GPU_BATCH_SIZE` in `config.py` |
| Ultralytics hangs on first run | Downloading `yolov8n.pt` with no internet access | Manually place a `.pt` file at `models/yolov8n.pt` |
| Folium map doesn't render | `streamlit-folium` missing | `pip install streamlit-folium` |
| Report never emails | SMTP env vars unset or wrong | Confirm `echo $SMTP_HOST` etc. in the same shell running Streamlit |
| Dashboard shows 0 detections | Thresholds too strict for the loaded weights | Lower sidebar sliders, especially on the default COCO-pretrained checkpoint |
| `torch.cuda.is_available()` is `False` on a GPU machine | CPU-only torch wheel installed | Reinstall with the correct `--index-url` for your CUDA version |

---

## 🗺️ Roadmap

- [ ] Real-time RTSP/video-stream ingestion (beyond static image folders)
- [ ] Fine-tuned infrastructure-specific YOLOv8 checkpoint bundled by default
- [ ] Historical trend charts (anomaly density over time, by geography)
- [ ] Role-based access for the dashboard
- [ ] Exportable GeoJSON alongside PDF reports

---
<img width="1069" height="781" alt="Screenshot 2026-07-10 090600" src="https://github.com/user-attachments/assets/71152a9c-c10a-46c7-a5d4-c3fcd3208555" />
<img width="1069" height="781" alt="Screenshot 2026-07-10 090600" src="https://github.com/user-attachments/assets/56836944-4aae-4263-b7e8-0bb49241eec1" />


## 🤝 Contributing

Issues and pull requests are welcome. Please keep changes modular — each file
in this project has a single, clear responsibility, and new features should
respect that boundary (detection logic in `inference.py`, persistence in
`database.py`, presentation in `app.py`, reporting in `reporter.py`).

## 📄 License

Released under the MIT License. See `LICENSE` for details.

## 🙏 Acknowledgments

Built on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics),
[PyTorch](https://pytorch.org/), [Streamlit](https://streamlit.io/), and
[ReportLab](https://www.reportlab.com/).

<div align="center">

*Built for edge deployment. No cloud required.*

</div>
