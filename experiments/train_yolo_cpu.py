#!/usr/bin/env python3
"""Reproduce the frozen CPU-only YOLO11n training run."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument(
        "--data", type=Path, default=Path("outputs/cpu_protocol_2800/joint.yaml")
    )
    parser.add_argument("--project", type=Path, default=Path("outputs"))
    parser.add_argument("--name", default="yolo11n_joint_cpu_final_seed2026")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(
            f"Protocol not found: {args.data}. Run build_cpu_yolo_protocol.py first."
        )
    if min(args.epochs, args.batch, args.imgsz, args.threads) < 1:
        raise ValueError("epochs, batch, imgsz, and threads must be positive")

    torch.set_num_threads(args.threads)
    model = YOLO(args.model)
    result = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device="cpu",
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=args.exist_ok,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.00125,
        lrf=0.05,
        weight_decay=5e-4,
        warmup_epochs=1.0,
        close_mosaic=2,
        seed=args.seed,
        deterministic=True,
        amp=False,
        val=False,
        plots=False,
        save=True,
        verbose=False,
    )
    run_dir = Path(result.save_dir)
    provenance = {
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "device": "cpu",
        "threads": args.threads,
        "data": str(args.data.resolve()),
        "model_source": args.model,
    }
    (run_dir / "reproduction.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "run_dir": str(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
