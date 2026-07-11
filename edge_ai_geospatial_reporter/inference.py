"""
inference.py - PyTorch/YOLOv8 detection pipeline for the Edge-AI Geospatial
Infrastructure & Asset Reporter.

Optimized for a 6GB VRAM budget (RTX 3050):
  * fp16 weights + torch.amp.autocast('cuda') mixed precision on GPU
  * small batch size, capped memory fraction (see config.py)
  * graceful fallback to multi-threaded CPU inference (Ryzen 7) when no
    CUDA device is present

Processes a folder of images as a stand-in for a live video feed, stamping
each frame with a simulated GPS coordinate (a straight-line drone/vehicle
flight path with small random jitter) since sample imagery typically has no
embedded EXIF GPS data.
"""

from __future__ import annotations

import logging
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import torch
from ultralytics import YOLO

import config
import database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("inference")

VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Detection:
    anomaly_type: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) in pixel coordinates
    latitude: float
    longitude: float
    source_image: str


@dataclass
class FlightPathSimulator:
    """Generates a plausible, monotonically-advancing lat/lon track so a
    static folder of images looks like a real inspection flight/drive when
    plotted on the dashboard map."""

    lat: float = config.SIM_GPS_CENTER_LAT
    lon: float = config.SIM_GPS_CENTER_LON
    step: float = config.SIM_GPS_STEP_DEGREES
    jitter: float = config.SIM_GPS_JITTER_DEGREES
    _heading: float = field(default_factory=lambda: random.uniform(0, 360))

    def next_coordinate(self) -> tuple:
        self._heading += random.uniform(-15, 15)
        rad = math.radians(self._heading)
        self.lat += self.step * math.cos(rad) + random.uniform(-self.jitter, self.jitter)
        self.lon += self.step * math.sin(rad) + random.uniform(-self.jitter, self.jitter)
        return round(self.lat, 6), round(self.lon, 6)


class ModelRegistry:
    """Lazily loads and caches the YOLO model so repeated pipeline runs
    (e.g. from Streamlit re-runs) don't reload weights from disk each time."""

    _model: Optional[YOLO] = None
    _device: Optional[str] = None

    @classmethod
    def get(cls) -> YOLO:
        if cls._model is None:
            weights = (
                str(config.MODEL_WEIGHTS_PATH)
                if config.MODEL_WEIGHTS_PATH.exists()
                else "yolov8n.pt"
            )
            logger.info("Loading YOLO weights from '%s' on device '%s'", weights, config.DEVICE)
            model = YOLO(weights)
            if config.CUDA_AVAILABLE:
                model.to(config.DEVICE)
                # fp16 weights roughly halve GPU memory footprint - important on 6GB cards.
                try:
                    model.model.half()
                except Exception:
                    logger.warning("Could not cast model to fp16; continuing in fp32.")
            cls._model = model
            cls._device = config.DEVICE
        return cls._model


def _map_class_name(raw_name: str) -> str:
    """Maps a raw model class name onto our infrastructure-anomaly vocabulary.

    The default checkpoint (yolov8n.pt) is COCO-pretrained and has no notion
    of potholes/cracks/etc. To keep the pipeline runnable out of the box we
    deterministically remap COCO classes onto the anomaly vocabulary so the
    rest of the system (thresholds, DB schema, map, report) can be exercised
    end-to-end. Swap in a fine-tuned checkpoint whose class names already
    match config.ANOMALY_CLASSES to skip this remapping entirely.
    """
    if raw_name in config.ANOMALY_CLASSES:
        return raw_name
    bucket = list(config.ANOMALY_CLASSES.keys())
    idx = abs(hash(raw_name)) % len(bucket)
    return bucket[idx]


def _passes_threshold(anomaly_type: str, confidence: float, overrides: Optional[dict] = None) -> bool:
    thresholds = overrides or config.CLASS_CONFIDENCE_THRESHOLDS
    threshold = thresholds.get(anomaly_type, config.DEFAULT_CONFIDENCE_THRESHOLD)
    return confidence >= threshold


def _run_gpu_inference(
    image_paths: List[Path],
    flight_sim: FlightPathSimulator,
    thresholds: Optional[dict] = None,
) -> List[Detection]:
    """Batched GPU inference using torch.amp.autocast('cuda') for mixed
    precision, keeping VRAM usage low enough for a 6GB card."""
    model = ModelRegistry.get()
    detections: List[Detection] = []
    batch_size = config.GPU_BATCH_SIZE

    for start in range(0, len(image_paths), batch_size):
        batch = image_paths[start : start + batch_size]
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            results = model.predict(
                source=[str(p) for p in batch],
                imgsz=config.INFERENCE_IMG_SIZE,
                device=config.DEVICE,
                quantize=16,
                verbose=False,
            )
        for image_path, result in zip(batch, results):
            lat, lon = flight_sim.next_coordinate()
            detections.extend(_extract_detections(result, image_path, lat, lon, thresholds))

        # Proactively release cached blocks between batches; on a 6GB card
        # this prevents fragmentation-driven OOMs during long runs.
        torch.cuda.empty_cache()

    return detections


def _run_cpu_inference(
    image_paths: List[Path],
    flight_sim: FlightPathSimulator,
    thresholds: Optional[dict] = None,
) -> List[Detection]:
    """Multi-threaded CPU fallback for machines without a CUDA device.
    Threads (rather than processes) are sufficient here because the actual
    per-image forward pass runs inside Ultralytics/PyTorch's C++/BLAS
    backend, which releases the GIL during the heavy lifting."""
    model = ModelRegistry.get()
    detections: List[Detection] = []

    # Pre-compute coordinates sequentially so the flight path stays coherent
    # even though inference itself runs across worker threads.
    coordinates = {p: flight_sim.next_coordinate() for p in image_paths}

    def _worker(image_path: Path) -> List[Detection]:
        with torch.no_grad():
            result = model.predict(
                source=str(image_path),
                imgsz=config.INFERENCE_IMG_SIZE,
                device="cpu",
                quantize=32,
                verbose=False,
            )[0]
        lat, lon = coordinates[image_path]
        return _extract_detections(result, image_path, lat, lon, thresholds)

    with ThreadPoolExecutor(max_workers=config.CPU_WORKER_THREADS) as executor:
        futures = {executor.submit(_worker, p): p for p in image_paths}
        for future in as_completed(futures):
            image_path = futures[future]
            try:
                detections.extend(future.result())
            except Exception as exc:
                logger.error("CPU inference failed for %s: %s", image_path, exc)

    return detections


def _extract_detections(
    result,
    image_path: Path,
    lat: float,
    lon: float,
    thresholds: Optional[dict] = None,
) -> List[Detection]:
    detections: List[Detection] = []
    if result.boxes is None:
        return detections

    names = result.names
    for box in result.boxes:
        raw_conf = float(box.conf[0])
        raw_class_idx = int(box.cls[0])
        raw_name = names.get(raw_class_idx, str(raw_class_idx))
        anomaly_type = _map_class_name(raw_name)

        if not _passes_threshold(anomaly_type, raw_conf, thresholds):
            continue

        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        detections.append(
            Detection(
                anomaly_type=anomaly_type,
                confidence=raw_conf,
                bbox=(x1, y1, x2, y2),
                latitude=lat,
                longitude=lon,
                source_image=str(image_path),
            )
        )
    return detections


def discover_feed_images(folder: Optional[Path] = None) -> List[Path]:
    """Lists all valid image files in the simulated feed folder, sorted so
    the flight-path simulation advances in a stable, repeatable order."""
    folder = folder or config.IMAGES_DIR
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES
    )


def run_inference_pipeline(
    image_folder: Optional[Path] = None,
    thresholds: Optional[dict] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Main entry point: runs detection over every image in `image_folder`
    (or the configured feed folder), persists all passing detections to the
    database, and returns a small run summary.

    `thresholds` may be a dict of {anomaly_type: min_confidence} to override
    config.CLASS_CONFIDENCE_THRESHOLDS for this run (used by the Streamlit
    sidebar sliders).
    `progress_callback(done, total)` is invoked as the run progresses so a
    UI can render a progress bar.
    """
    started = time.time()
    image_paths = discover_feed_images(image_folder)
    flight_sim = FlightPathSimulator()

    if not image_paths:
        logger.warning("No images found in feed folder %s", image_folder or config.IMAGES_DIR)
        return {
            "images_processed": 0,
            "detections_found": 0,
            "device_used": config.DEVICE,
            "elapsed_seconds": 0.0,
        }

    logger.info(
        "Starting inference run: %d images, device=%s, fp16=%s",
        len(image_paths),
        config.DEVICE,
        config.USE_FP16,
    )

    if config.CUDA_AVAILABLE:
        detections = _run_gpu_inference(image_paths, flight_sim, thresholds)
    else:
        detections = _run_cpu_inference(image_paths, flight_sim, thresholds)

    if progress_callback:
        progress_callback(len(image_paths), len(image_paths))

    records = [
        {
            "anomaly_type": d.anomaly_type,
            "confidence": d.confidence,
            "latitude": d.latitude,
            "longitude": d.longitude,
            "source_image": d.source_image,
            "bbox": d.bbox,
        }
        for d in detections
    ]
    inserted = database.insert_detections_bulk(records)

    elapsed = time.time() - started
    logger.info(
        "Inference run complete: %d images, %d detections stored in %.2fs",
        len(image_paths),
        inserted,
        elapsed,
    )

    return {
        "images_processed": len(image_paths),
        "detections_found": inserted,
        "device_used": config.DEVICE,
        "elapsed_seconds": round(elapsed, 2),
    }


if __name__ == "__main__":
    summary = run_inference_pipeline()
    print(summary)
