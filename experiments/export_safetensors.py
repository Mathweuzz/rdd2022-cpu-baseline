#!/usr/bin/env python3
"""Export a trusted Ultralytics pickle checkpoint as safe tensors plus YAML."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != ".safetensors":
        parser.error("--output must end in .safetensors")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    # weights_only=False is required by legacy Ultralytics checkpoints. Only run
    # this exporter on a checkpoint produced locally or obtained from a trusted
    # source; consumers should use the resulting safetensors file instead.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    network = checkpoint.get("ema") or checkpoint.get("model")
    if network is None or not hasattr(network, "state_dict"):
        raise ValueError("checkpoint does not contain an Ultralytics model")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in network.state_dict().items()
    }
    save_file(
        state,
        str(args.output),
        metadata={"format": "pt", "task": "detect", "classes": "D00,D10,D20,D40"},
    )
    config = args.output.with_suffix(".yaml")
    config.write_text(yaml.safe_dump(network.yaml, sort_keys=False), encoding="utf-8")
    print(f"{sha256(args.output)}  {args.output}")
    print(f"{sha256(config)}  {config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
