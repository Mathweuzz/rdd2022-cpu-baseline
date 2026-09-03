#!/usr/bin/env python3
"""Qualify or train RT-DETR-R50 on the cleaned RDD2022 COCO split."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from train_torchvision_baseline import CocoDetectionDataset, select_smoke_indices


CLASS_NAMES = ("D00", "D10", "D20", "D40")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/rdd2022/clean"))
    parser.add_argument("--output", type=Path, default=Path("outputs/rtdetr_smoke"))
    parser.add_argument("--checkpoint", default="PekingU/rtdetr_r50vd")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--smoke-images", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=1)
    parser.add_argument("--max-val-images", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--save-checkpoint", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--negative-only",
        action="store_true",
        help="select only images without target annotations for a loss diagnostic",
    )
    return parser.parse_args()


def collate_with_processor(processor: RTDetrImageProcessor):
    def collate(batch: list[tuple[torch.Tensor, dict]]) -> dict:
        images = []
        annotations = []
        for image, target in batch:
            objects = []
            for box, label, area, crowd in zip(
                target["boxes"], target["labels"], target["area"], target["iscrowd"], strict=True
            ):
                xmin, ymin, xmax, ymax = (float(value) for value in box)
                objects.append(
                    {
                        "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                        "category_id": int(label) - 1,
                        "area": float(area),
                        "iscrowd": int(crowd),
                    }
                )
            images.append(image)
            annotations.append({"image_id": int(target["image_id"]), "annotations": objects})
        return processor(images=images, annotations=annotations, return_tensors="pt")

    return collate


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if key == "labels":
            moved[key] = [
                {label_key: label_value.to(device) for label_key, label_value in item.items()}
                for item in value
            ]
        else:
            moved[key] = value.to(device)
    return moved


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto"
        else "cpu"
    )
    args.output.mkdir(parents=True, exist_ok=True)

    id2label = {index: name for index, name in enumerate(CLASS_NAMES)}
    label2id = {name: index for index, name in id2label.items()}
    processor = RTDetrImageProcessor.from_pretrained(
        args.checkpoint,
        size={"height": args.image_size, "width": args.image_size},
        local_files_only=args.local_files_only,
    )
    model = RTDetrForObjectDetection.from_pretrained(
        args.checkpoint,
        num_labels=len(CLASS_NAMES),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        local_files_only=args.local_files_only,
    ).to(device)

    train_full = CocoDetectionDataset(args.dataset, "train", augment=False)
    if args.negative_only:
        negative_indices = [
            index for index, record in enumerate(train_full.images)
            if not train_full.annotations.get(int(record["id"]))
        ]
        random.Random(args.seed).shuffle(negative_indices)
        indices = negative_indices[: min(args.smoke_images, len(negative_indices))]
        if not indices:
            raise RuntimeError("no negative images are available")
    else:
        indices = select_smoke_indices(
            train_full, min(args.smoke_images, len(train_full)), args.seed
        )
    loader = DataLoader(
        Subset(train_full, indices), batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_with_processor(processor)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    losses = []
    started = time.perf_counter()
    model.train()
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_train_batches:
            break
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        if not torch.isfinite(output.loss):
            raise RuntimeError(f"non-finite loss: {output.loss}")
        output.loss.backward()
        gradient_l1 = sum(
            float(parameter.grad.detach().abs().sum())
            for parameter in model.parameters() if parameter.grad is not None
        )
        optimizer.step()
        losses.append(
            {
                "batch": batch_index,
                "total": float(output.loss.detach()),
                "gradient_l1": gradient_l1,
                "components": {key: float(value.detach()) for key, value in output.loss_dict.items()},
            }
        )
    training_seconds = time.perf_counter() - started

    checkpoint_reload = False
    if args.save_checkpoint:
        checkpoint_dir = args.output / "checkpoint"
        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        reloaded = RTDetrForObjectDetection.from_pretrained(
            checkpoint_dir, local_files_only=True
        ).to(device)
        checkpoint_reload = True
    else:
        reloaded = model

    val_dataset = CocoDetectionDataset(args.dataset, "val", augment=False)
    prediction_count = 0
    predictions = []
    image_ids = []
    inference_seconds = 0.0
    reloaded.eval()
    with torch.inference_mode():
        for index in range(min(args.max_val_images, len(val_dataset))):
            image, target = val_dataset[index]
            image_id = int(target["image_id"])
            image_ids.append(image_id)
            encoded = processor(images=[image], return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            inference_started = time.perf_counter()
            output = reloaded(**encoded)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - inference_started
            height, width = (int(value) for value in target["orig_size"])
            detections = processor.post_process_object_detection(
                output, threshold=0.0,
                target_sizes=torch.tensor([[height, width]], device=device),
            )[0]
            order = torch.argsort(detections["scores"], descending=True)[:100]
            prediction_count += len(order)
            for detection_index in order:
                xmin, ymin, xmax, ymax = detections["boxes"][detection_index].tolist()
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": int(detections["labels"][detection_index]) + 1,
                        "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                        "score": float(detections["scores"][detection_index]),
                    }
                )

    coco_metrics = None
    if image_ids and predictions:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        annotation_path = args.dataset / "annotations" / "instances_val.json"
        ground_truth = COCO(str(annotation_path))
        detections = ground_truth.loadRes(predictions)
        evaluator = COCOeval(ground_truth, detections, iouType="bbox")
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        metric_names = (
            "map_50_95", "ap50", "ap75", "ap_small", "ap_medium", "ap_large",
            "ar_1", "ar_10", "ar_100", "ar_small", "ar_medium", "ar_large",
        )
        coco_metrics = {
            name: float(value) for name, value in zip(metric_names, evaluator.stats, strict=True)
        }

    summary = {
        "status": "diagnostic_only", "model": "RT-DETR-R50",
        "source_checkpoint": args.checkpoint, "device": str(device),
        "image_size": args.image_size, "seed": args.seed,
        "negative_only": args.negative_only,
        "train_images_available": len(train_full), "smoke_indices": indices,
        "losses": losses, "training_seconds": training_seconds,
        "validation_images": min(args.max_val_images, len(val_dataset)),
        "prediction_count_capped_at_100_per_image": prediction_count,
        "coco_metrics": coco_metrics,
        "inference_seconds": inference_seconds, "checkpoint_reload": checkpoint_reload,
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
