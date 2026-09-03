#!/usr/bin/env python3
"""Build deterministic, domain-balanced YOLO lists for CPU-budgeted experiments."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


DOMAINS = (
    "China_Drone", "China_MotorBike", "Czech", "India", "Japan", "Norway",
    "United_States",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/rdd2022/clean"))
    parser.add_argument("--output", type=Path, default=Path("outputs/cpu_protocol"))
    parser.add_argument("--train-images", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def allocate(total: int, domains: list[str]) -> dict[str, int]:
    base, remainder = divmod(total, len(domains))
    return {domain: base + (position < remainder) for position, domain in enumerate(domains)}


def sample_domain(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    negatives = [row for row in rows if int(row["negatives"])]
    positives = [row for row in rows if not int(row["negatives"])]
    negative_count = round(count * len(negatives) / max(len(rows), 1))
    negative_count = min(negative_count, len(negatives))
    positive_count = min(count - negative_count, len(positives))
    remaining = count - negative_count - positive_count
    if remaining:
        extra_negatives = min(remaining, len(negatives) - negative_count)
        negative_count += extra_negatives
        remaining -= extra_negatives
        positive_count += min(remaining, len(positives) - positive_count)
    return rng.sample(negatives, negative_count) + rng.sample(positives, positive_count)


def image_path(dataset: Path, row: dict) -> str:
    return str((dataset / "images" / row["clean_split"] / row["clean_file"]).resolve())


def write_list(path: Path, dataset: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(image_path(dataset, row) for row in rows) + "\n", encoding="utf-8")


def write_yaml(path: Path, train: Path, val: Path, test: Path) -> None:
    payload = (
        f"train: {train.resolve()}\n"
        f"val: {val.resolve()}\n"
        f"test: {test.resolve()}\n"
        "names:\n  0: D00\n  1: D10\n  2: D20\n  3: D40\n"
    )
    path.write_text(payload, encoding="utf-8")


def summarize(rows: list[dict]) -> dict:
    return {
        "images": len(rows),
        "domains": dict(sorted(Counter(row["domain"] for row in rows).items())),
        "negative_images": sum(int(row["negatives"]) for row in rows),
        "instances": {
            name: sum(int(row[name]) for row in rows) for name in ("D00", "D10", "D20", "D40")
        },
    }


def main() -> int:
    args = parse_args()
    if args.train_images < len(DOMAINS):
        raise ValueError("train-images must be at least the number of domains")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.dataset / "split_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labeled = [row for row in rows if row["clean_split"] in {"train", "val", "test"}]
    by_split_domain: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in labeled:
        by_split_domain[(row["clean_split"], row["domain"])].append(row)

    registry = {}
    experiments = [("joint", None)] + [(f"lodo_{domain}", domain) for domain in DOMAINS]
    for name, heldout in experiments:
        sources = [domain for domain in DOMAINS if domain != heldout]
        quotas = allocate(args.train_images, sources)
        rng = random.Random(f"{args.seed}:{name}")
        train_rows = []
        for domain in sources:
            train_rows.extend(sample_domain(by_split_domain[("train", domain)], quotas[domain], rng))
        rng.shuffle(train_rows)
        val_rows = [
            row for domain in sources for row in by_split_domain[("val", domain)]
        ]
        test_domains = [heldout] if heldout else list(DOMAINS)
        test_rows = [
            row for domain in test_domains for row in by_split_domain[("test", domain)]
        ]
        train_path = args.output / f"{name}_train.txt"
        val_path = args.output / f"{name}_val.txt"
        test_path = args.output / f"{name}_test.txt"
        yaml_path = args.output / f"{name}.yaml"
        write_list(train_path, args.dataset, train_rows)
        write_list(val_path, args.dataset, val_rows)
        write_list(test_path, args.dataset, test_rows)
        write_yaml(yaml_path, train_path, val_path, test_path)
        registry[name] = {
            "heldout_domain": heldout,
            "train": summarize(train_rows),
            "validation": summarize(val_rows),
            "test": summarize(test_rows),
            "yaml": str(yaml_path.resolve()),
        }

    (args.output / "protocol.json").write_text(
        json.dumps({"seed": args.seed, "train_images": args.train_images, "runs": registry}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({name: item["train"] for name, item in registry.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
