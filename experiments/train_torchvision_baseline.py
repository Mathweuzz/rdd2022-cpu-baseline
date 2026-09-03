#!/usr/bin/env python3
"""Train or smoke-test TorchVision detectors on cleaned RDD2022."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    retinanet_resnet50_fpn_v2,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


class CocoDetectionDataset(Dataset):
    """Minimal COCO reader that does not depend on pycocotools."""

    def __init__(self, root: Path, split: str, augment: bool = False) -> None:
        self.root = root
        self.split = split
        self.augment = augment
        annotation_path = root / "annotations" / f"instances_{split}.json"
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        self.images = sorted(payload["images"], key=lambda item: item["id"])
        self.annotations: dict[int, list[dict]] = defaultdict(list)
        for annotation in payload["annotations"]:
            self.annotations[int(annotation["image_id"])].append(annotation)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        record = self.images[index]
        image_path = self.root / "images" / self.split / record["file_name"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size

        annotations = self.annotations.get(int(record["id"]), [])
        boxes = []
        labels = []
        areas = []
        crowds = []
        for annotation in annotations:
            xmin, ymin, box_width, box_height = annotation["bbox"]
            boxes.append([xmin, ymin, xmin + box_width, ymin + box_height])
            labels.append(int(annotation["category_id"]))
            areas.append(float(annotation["area"]))
            crowds.append(int(annotation.get("iscrowd", 0)))

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        image_tensor = F.pil_to_tensor(image)
        image_tensor = F.convert_image_dtype(image_tensor, torch.float32)

        if self.augment and random.random() < 0.5:
            image_tensor = torch.flip(image_tensor, dims=(-1,))
            if len(boxes_tensor):
                old_xmin = boxes_tensor[:, 0].clone()
                old_xmax = boxes_tensor[:, 2].clone()
                boxes_tensor[:, 0] = width - old_xmax
                boxes_tensor[:, 2] = width - old_xmin

        target = {
            "boxes": boxes_tensor,
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(int(record["id"]), dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(crowds, dtype=torch.int64),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
        }
        return image_tensor, target


def collate_detection(batch: list[tuple[torch.Tensor, dict]]) -> tuple[list, list]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def select_smoke_indices(
    dataset: CocoDetectionDataset,
    count: int,
    seed: int,
    eligible: list[int] | None = None,
) -> list[int]:
    candidates = list(range(len(dataset))) if eligible is None else list(eligible)
    if count >= len(candidates):
        return candidates
    positive = [
        index
        for index in candidates
        for record in [dataset.images[index]]
        if dataset.annotations.get(int(record["id"]))
    ]
    negative = [
        index
        for index in candidates
        for record in [dataset.images[index]]
        if not dataset.annotations.get(int(record["id"]))
    ]
    rng = random.Random(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)
    positive_count = min(len(positive), max(1, (3 * count) // 4))
    negative_count = min(len(negative), count - positive_count)
    remaining = count - positive_count - negative_count
    if remaining:
        additional_positive = min(len(positive) - positive_count, remaining)
        positive_count += additional_positive
        remaining -= additional_positive
        negative_count += min(len(negative) - negative_count, remaining)
    chosen = positive[:positive_count] + negative[:negative_count]
    rng.shuffle(chosen)
    return chosen


def move_target(target: dict[str, torch.Tensor], device: torch.device) -> dict:
    return {key: value.to(device) for key, value in target.items() if key != "orig_size"}


def evaluate_coco(
    model: torch.nn.Module,
    dataset: CocoDetectionDataset,
    indices: list[int],
    device: torch.device,
) -> tuple[dict[str, float], float, int]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise RuntimeError(
            "pycocotools is required for evaluation; install requirements-experiments.txt"
        ) from error

    predictions = []
    image_ids = []
    inference_seconds = 0.0
    model.eval()
    with torch.inference_mode():
        for index in indices:
            image, _ = dataset[index]
            record = dataset.images[index]
            image_id = int(record["id"])
            image_ids.append(image_id)
            started = time.perf_counter()
            output = model([image.to(device)])[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - started
            boxes = output["boxes"].detach().cpu()
            labels = output["labels"].detach().cpu()
            scores = output["scores"].detach().cpu()
            order = torch.argsort(scores, descending=True)[:100]
            for box, label, score in zip(boxes[order], labels[order], scores[order]):
                xmin, ymin, xmax, ymax = box.tolist()
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                        "score": float(score),
                    }
                )

    annotation_path = dataset.root / "annotations" / f"instances_{dataset.split}.json"
    ground_truth = COCO(str(annotation_path))
    detections = ground_truth.loadRes(predictions)
    evaluator = COCOeval(ground_truth, detections, iouType="bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = (
        "map_50_95",
        "ap50",
        "ap75",
        "ap_small",
        "ap_medium",
        "ap_large",
        "ar_1",
        "ar_10",
        "ar_100",
        "ar_small",
        "ar_medium",
        "ar_large",
    )
    metrics = {name: float(value) for name, value in zip(names, evaluator.stats)}
    return metrics, inference_seconds, len(predictions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/rdd2022/clean"))
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline_smoke"))
    parser.add_argument(
        "--model",
        default="retinanet_resnet50_fpn_v2",
        choices=(
            "fasterrcnn_mobilenet_v3_large_320_fpn",
            "retinanet_resnet50_fpn_v2",
            "ssdlite320_mobilenet_v3_large",
        ),
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=1)
    parser.add_argument("--smoke-images", type=int, default=8)
    parser.add_argument("--max-val-images", type=int, default=8)
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--exclude-domain",
        help="exclude one domain from training and source-domain validation",
    )
    parser.add_argument(
        "--evaluation-domain",
        help="evaluate only one domain within the selected evaluation split",
    )
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--min-size", type=int, default=320)
    parser.add_argument("--max-size", type=int, default=640)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="request deterministic PyTorch algorithms",
    )
    parser.add_argument(
        "--disable-augmentation",
        action="store_true",
        help="disable horizontal flipping (recommended for overfit diagnostics)",
    )
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="use ImageNet MobileNetV3 weights (may download weights)",
    )
    parser.add_argument(
        "--pretrained-detector",
        action="store_true",
        help="initialize Faster R-CNN from official COCO detection weights",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(args.deterministic)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = not args.deterministic
        torch.backends.cudnn.deterministic = args.deterministic

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if args.device != "auto"
        else "cpu"
    )
    args.output.mkdir(parents=True, exist_ok=True)

    train_full = CocoDetectionDataset(
        args.dataset, "train", augment=not args.disable_augmentation
    )
    val_dataset = CocoDetectionDataset(args.dataset, args.eval_split, augment=False)
    train_eligible = [
        index for index, record in enumerate(train_full.images)
        if not args.exclude_domain
        or Path(record["file_name"]).parts[0] != args.exclude_domain
    ]
    smoke_count = min(args.smoke_images, len(train_eligible))
    train_dataset = Subset(
        train_full,
        select_smoke_indices(train_full, smoke_count, args.seed, train_eligible),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collate_detection,
    )

    if args.model == "retinanet_resnet50_fpn_v2":
        backbone_weights = ResNet50_Weights.DEFAULT if args.pretrained_backbone else None
        model = retinanet_resnet50_fpn_v2(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=5,
            min_size=args.min_size,
            max_size=args.max_size,
        )
    elif args.model == "ssdlite320_mobilenet_v3_large":
        backbone_weights = (
            MobileNet_V3_Large_Weights.DEFAULT if args.pretrained_backbone else None
        )
        model = ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=5,
        )
    else:
        if args.pretrained_detector:
            model = fasterrcnn_mobilenet_v3_large_320_fpn(
                weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
                min_size=args.min_size,
                max_size=args.max_size,
            )
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)
        else:
            backbone_weights = (
                MobileNet_V3_Large_Weights.DEFAULT if args.pretrained_backbone else None
            )
            model = fasterrcnn_mobilenet_v3_large_320_fpn(
                weights=None,
                weights_backbone=backbone_weights,
                num_classes=5,
                min_size=args.min_size,
                max_size=args.max_size,
            )
    model = model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
    )
    resumed_from = None
    if args.resume is not None:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        saved_model = saved.get("config", {}).get("model")
        if saved_model is not None and saved_model != args.model:
            raise RuntimeError(
                f"checkpoint model {saved_model!r} does not match {args.model!r}"
            )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        resumed_from = str(args.resume)

    started = time.perf_counter()
    train_losses: list[dict[str, float]] = []
    processed_batches = 0
    model.train()
    for epoch in range(args.epochs):
        for images, targets in loader:
            batch_started = time.perf_counter()
            images = [image.to(device) for image in images]
            targets = [move_target(target, device) for target in targets]
            loss_dict = model(images, targets)
            total_loss = sum(loss_dict.values())
            if not torch.isfinite(total_loss):
                raise RuntimeError(f"non-finite loss: {float(total_loss)}")
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            record = {key: float(value.detach().cpu()) for key, value in loss_dict.items()}
            record["total"] = float(total_loss.detach().cpu())
            record["seconds"] = time.perf_counter() - batch_started
            record["epoch"] = epoch
            record["batch"] = processed_batches
            train_losses.append(record)
            processed_batches += 1
            if args.log_every and processed_batches % args.log_every == 0:
                print(json.dumps(record, sort_keys=True), flush=True)
            if args.max_train_batches and processed_batches >= args.max_train_batches:
                break
        if args.max_train_batches and processed_batches >= args.max_train_batches:
            break

    validation_eligible = [
        index for index, record in enumerate(val_dataset.images)
        if (
            (not args.exclude_domain or Path(record["file_name"]).parts[0] != args.exclude_domain)
            and (
                not args.evaluation_domain
                or Path(record["file_name"]).parts[0] == args.evaluation_domain
            )
        )
    ]
    validation_indices = select_smoke_indices(
        val_dataset,
        min(args.max_val_images, len(validation_eligible)),
        args.seed,
        validation_eligible,
    )
    coco_metrics, inference_seconds, prediction_count = evaluate_coco(
        model, val_dataset, validation_indices, device
    )

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "classes": ["background", "D00", "D10", "D20", "D40"],
        "config": vars(args) | {"dataset": str(args.dataset), "output": str(args.output)},
        "train_losses": train_losses,
    }
    checkpoint_path = args.output / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    summary = {
        "status": "passed",
        "purpose": "integration smoke test; not a scientific result",
        "model": args.model,
        "device": str(device),
        "pretrained_backbone": args.pretrained_backbone,
        "pretrained_detector": args.pretrained_detector,
        "resumed_from": resumed_from,
        "seed": args.seed,
        "threads": args.threads,
        "train_images_available": len(train_full),
        "smoke_images_selected": len(train_dataset),
        "train_batches": processed_batches,
        "train_losses": train_losses,
        "validation_images": len(validation_indices),
        "inference_seconds_total": inference_seconds,
        "inference_seconds_per_image": inference_seconds / len(validation_indices),
        "prediction_count": prediction_count,
        "coco_metrics_diagnostic": coco_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "checkpoint": str(checkpoint_path),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
