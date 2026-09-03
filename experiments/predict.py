#!/usr/bin/env python3
"""Run low-memory YOLO inference on one image or an image directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model_io import load_yolo


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def image_paths(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    if source.is_dir():
        return sorted(
            path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
    raise FileNotFoundError(f"No supported image source found at {source}")


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("conf must be between zero and one")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    model = load_yolo(args.model, args.config)
    records = []
    for index, source in enumerate(image_paths(args.source)):
        result = model.predict(
            source=str(source), device="cpu", imgsz=args.imgsz, conf=args.conf,
            batch=1, workers=0, verbose=False,
        )[0]
        rendered = args.output / f"{index:05d}_{source.name}"
        result.save(filename=str(rendered))
        boxes = result.boxes
        detections = []
        if boxes is not None:
            for xyxy, class_index, confidence in zip(
                boxes.xyxy.cpu(), boxes.cls.cpu(), boxes.conf.cpu(), strict=True
            ):
                detections.append({
                    "class_id": int(class_index),
                    "class_name": result.names[int(class_index)],
                    "confidence": float(confidence),
                    "xyxy": [float(value) for value in xyxy],
                })
        records.append({
            "source": str(source), "rendered": str(rendered),
            "detections": detections,
        })
    summary = args.output / "predictions.json"
    summary.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(records), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
