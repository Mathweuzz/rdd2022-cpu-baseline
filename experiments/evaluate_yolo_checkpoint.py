#!/usr/bin/env python3
"""Run one YOLO inference pass and compute aggregate and sliced COCO metrics."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import ultralytics
from pycocotools.coco import COCO
from ultralytics import YOLO

from evaluate_cpu_checkpoint import coco_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/rdd2022/clean"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    annotation_path = args.dataset / "annotations" / f"instances_{args.split}.json"
    coco = COCO(str(annotation_path))
    records = sorted(coco.dataset["images"], key=lambda item: item["id"])
    if args.max_images:
        records = records[: args.max_images]
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco.dataset["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    model = YOLO(str(args.model))
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    started = time.perf_counter()
    predictions: list[dict] = []
    domain_ids: dict[str, list[int]] = defaultdict(list)
    negative_ids: set[int] = set()
    threshold_counts: dict[int, int] = {}
    speed_ms: dict[str, list[float]] = defaultdict(list)
    # Deliberately invoke inference on one pathname at a time.  Passing the full
    # pathname list to Ultralytics caused its source loader to retain decoded
    # images and exhaust RAM even when stream=True and batch=1.
    for position, record in enumerate(records, start=1):
        source = str(
            (args.dataset / "images" / args.split / record["file_name"]).resolve()
        )
        result = model.predict(
            source=source, stream=False, imgsz=args.imgsz, batch=1, device="cpu",
            conf=0.001, iou=0.7, max_det=100, verbose=False, workers=0,
        )[0]
        image_id = int(record["id"])
        domain_ids[Path(record["file_name"]).parts[0]].append(image_id)
        if not annotations_by_image[image_id]:
            negative_ids.add(image_id)
        boxes = result.boxes
        scores = boxes.conf.detach().cpu() if boxes is not None else torch.empty(0)
        threshold_counts[image_id] = int((scores >= args.score_threshold).sum())
        if boxes is not None:
            for xyxy, class_index, score in zip(
                boxes.xyxy.detach().cpu(), boxes.cls.detach().cpu(), scores, strict=True
            ):
                xmin, ymin, xmax, ymax = xyxy.tolist()
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(class_index) + 1,
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "score": float(score),
                })
        for name, milliseconds in result.speed.items():
            speed_ms[name].append(float(milliseconds))
        del result, boxes, scores
        if position % 100 == 0:
            gc.collect()
        if position % 250 == 0:
            print(json.dumps({
                "evaluated": position,
                "total": len(records),
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }), flush=True)

    # Persist the only large result collection before releasing inference state.
    # This makes an interrupted low-memory run recoverable and avoids retaining
    # model tensors while the repeated COCO slices are evaluated.
    (args.output / "predictions.json").write_text(
        json.dumps(predictions) + "\n", encoding="utf-8"
    )
    del model
    gc.collect()

    image_ids = [int(record["id"]) for record in records]
    metrics = coco_metrics(coco, predictions, image_ids)
    per_class = {}
    for category in coco.dataset["categories"]:
        per_class[category["name"]] = coco_metrics(
            coco, predictions, image_ids, cat_ids=[int(category["id"])]
        )
    per_domain = {
        domain: coco_metrics(coco, predictions, ids)
        for domain, ids in sorted(domain_ids.items())
    }
    domain_map = {domain: item["map_50_95"] for domain, item in per_domain.items()}
    summary = {
        "status": "complete",
        "model": str(args.model),
        "split": args.split,
        "images": len(records),
        "instances": len(coco.dataset["annotations"]),
        "parameters": parameter_count,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "metrics": metrics,
        "per_class": per_class,
        "per_domain": per_domain,
        "domain_macro_map_50_95": float(np.mean(list(domain_map.values()))),
        "domain_worst_map_50_95": float(min(domain_map.values())),
        "domain_worst": min(domain_map, key=domain_map.get),
        "negative_images": len(negative_ids),
        "negative_false_positive_rate_at_threshold": (
            sum(threshold_counts[item] > 0 for item in negative_ids) / len(negative_ids)
        ),
        "mean_detections_on_negatives_at_threshold": float(
            np.mean([threshold_counts[item] for item in negative_ids])
        ),
        "score_threshold": args.score_threshold,
        "mean_speed_ms": {name: float(np.mean(values)) for name, values in speed_ms.items()},
        "elapsed_seconds": time.perf_counter() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "platform": platform.platform(),
        },
    }
    (args.output / "evaluation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
