#!/usr/bin/env python3
"""Verify whether candidate detectors learn from an all-negative RDD2022 batch."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    ssdlite320_mobilenet_v3_large,
)

from train_torchvision_baseline import CocoDetectionDataset, move_target


def evaluate_model(name: str, model: torch.nn.Module, images: list, targets: list) -> dict:
    model.train()
    model.zero_grad(set_to_none=True)
    started = time.perf_counter()
    loss_dict = model(images, targets)
    total = sum(loss_dict.values())
    total.backward()
    elapsed = time.perf_counter() - started
    gradient_l1 = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    return {
        "model": name,
        "losses": {key: float(value.detach()) for key, value in loss_dict.items()},
        "total_loss": float(total.detach()),
        "gradient_l1": gradient_l1,
        "seconds": elapsed,
        "learns_from_negative_only_batch": bool(total.detach() > 0 and gradient_l1 > 0),
    }


def main() -> int:
    torch.manual_seed(2026)
    torch.set_num_threads(8)
    dataset = CocoDetectionDataset(Path("data/rdd2022/clean"), "train", augment=False)
    negative_indices = [
        index
        for index, record in enumerate(dataset.images)
        if not dataset.annotations.get(int(record["id"]))
        and record["domain"] == "Czech"
    ][:2]
    if len(negative_indices) != 2:
        raise RuntimeError("could not select two Czech negative images")
    batch = [dataset[index] for index in negative_indices]
    images = [image for image, _ in batch]
    targets = [move_target(target, torch.device("cpu")) for _, target in batch]
    sample_files = [dataset.images[index]["file_name"] for index in negative_indices]

    builders = (
        (
            "ssdlite320_mobilenet_v3_large",
            lambda: ssdlite320_mobilenet_v3_large(
                weights=None, weights_backbone=None, num_classes=5
            ),
        ),
        (
            "retinanet_resnet50_fpn_v2",
            lambda: retinanet_resnet50_fpn_v2(
                weights=None,
                weights_backbone=None,
                num_classes=5,
                min_size=320,
                max_size=640,
            ),
        ),
    )
    results = []
    for name, builder in builders:
        model = builder()
        results.append(evaluate_model(name, model, images, targets))
        del model
        gc.collect()

    payload = {
        "status": "passed",
        "purpose": "candidate selection diagnostic; not a scientific result",
        "seed": 2026,
        "images": sample_files,
        "ground_truth_boxes": [len(target["boxes"]) for target in targets],
        "results": results,
    }
    output = Path("outputs/negative_batch_test")
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not results[1]["learns_from_negative_only_batch"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
