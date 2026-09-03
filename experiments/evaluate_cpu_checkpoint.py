#!/usr/bin/env python3
"""Evaluate a trained TorchVision checkpoint once and report sliced COCO metrics."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    retinanet_resnet50_fpn_v2,
    ssdlite320_mobilenet_v3_large,
)

from train_torchvision_baseline import CocoDetectionDataset


METRIC_NAMES = (
    "map_50_95", "ap50", "ap75", "ap_small", "ap_medium", "ap_large",
    "ar_1", "ar_10", "ar_100", "ar_small", "ar_medium", "ar_large",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/rdd2022/clean"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--domain", help="evaluate only this acquisition domain")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def build_model(name: str, min_size: int, max_size: int) -> torch.nn.Module:
    if name == "fasterrcnn_mobilenet_v3_large_320_fpn":
        return fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None, weights_backbone=None, num_classes=5,
            min_size=min_size, max_size=max_size,
        )
    if name == "retinanet_resnet50_fpn_v2":
        return retinanet_resnet50_fpn_v2(
            weights=None, weights_backbone=None, num_classes=5,
            min_size=min_size, max_size=max_size,
        )
    if name == "ssdlite320_mobilenet_v3_large":
        return ssdlite320_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=5,
        )
    raise ValueError(f"unsupported model: {name}")


def coco_metrics(coco: COCO, predictions: list[dict], image_ids: list[int], cat_ids=None) -> dict:
    if not predictions:
        return {name: 0.0 for name in METRIC_NAMES}
    detections = coco.loadRes(predictions)
    evaluator = COCOeval(coco, detections, iouType="bbox")
    evaluator.params.imgIds = image_ids
    if cat_ids is not None:
        evaluator.params.catIds = cat_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {name: float(value) for name, value in zip(METRIC_NAMES, evaluator.stats)}


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = saved["config"]
    model = build_model(
        config["model"], int(config.get("min_size", 320)), int(config.get("max_size", 320))
    )
    model.load_state_dict(saved["model"])
    model.eval()

    dataset = CocoDetectionDataset(args.dataset, args.split, augment=False)
    indices = [
        index for index, record in enumerate(dataset.images)
        if not args.domain or Path(record["file_name"]).parts[0] == args.domain
    ]
    if args.max_images:
        indices = indices[: args.max_images]

    predictions: list[dict] = []
    image_ids: list[int] = []
    domain_ids: dict[str, list[int]] = defaultdict(list)
    negative_ids: set[int] = set()
    positive_ids: set[int] = set()
    high_score_counts: dict[int, int] = {}
    inference_times: list[float] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for position, index in enumerate(indices, start=1):
            image, target = dataset[index]
            record = dataset.images[index]
            image_id = int(record["id"])
            domain = Path(record["file_name"]).parts[0]
            image_ids.append(image_id)
            domain_ids[domain].append(image_id)
            (positive_ids if len(target["boxes"]) else negative_ids).add(image_id)
            item_started = time.perf_counter()
            output = model([image])[0]
            inference_times.append(time.perf_counter() - item_started)
            scores = output["scores"].detach().cpu()
            order = torch.argsort(scores, descending=True)[:100]
            high_score_counts[image_id] = int((scores[order] >= args.score_threshold).sum())
            for box, label, score in zip(
                output["boxes"].detach().cpu()[order],
                output["labels"].detach().cpu()[order], scores[order], strict=True,
            ):
                xmin, ymin, xmax, ymax = box.tolist()
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "score": float(score),
                })
            if position % 250 == 0:
                print(json.dumps({"evaluated": position, "total": len(indices)}), flush=True)

    annotation_path = args.dataset / "annotations" / f"instances_{args.split}.json"
    coco = COCO(str(annotation_path))
    results = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "model": config["model"],
        "split": args.split,
        "images": len(image_ids),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "metrics": coco_metrics(coco, predictions, image_ids),
        "per_class": {},
        "per_domain": {},
        "negative_images": len(negative_ids),
        "positive_images": len(positive_ids),
        "negative_false_positive_rate_at_threshold": (
            sum(high_score_counts[item] > 0 for item in negative_ids) / len(negative_ids)
            if negative_ids else 0.0
        ),
        "mean_high_score_detections_on_negatives": (
            float(np.mean([high_score_counts[item] for item in negative_ids]))
            if negative_ids else 0.0
        ),
        "score_threshold": args.score_threshold,
        "inference_seconds_total": float(sum(inference_times)),
        "inference_seconds_per_image_median": float(np.median(inference_times)),
        "inference_seconds_per_image_mean": float(np.mean(inference_times)),
        "elapsed_seconds": time.perf_counter() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "platform": platform.platform(),
        },
    }
    categories = {item["id"]: item["name"] for item in coco.dataset["categories"]}
    for category_id, name in categories.items():
        results["per_class"][name] = coco_metrics(
            coco, predictions, image_ids, cat_ids=[category_id]
        )
    for domain, ids in sorted(domain_ids.items()):
        results["per_domain"][domain] = coco_metrics(coco, predictions, ids)

    (args.output / "predictions.json").write_text(
        json.dumps(predictions) + "\n", encoding="utf-8"
    )
    (args.output / "evaluation.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
