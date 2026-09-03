#!/usr/bin/env python3
"""Prepara o RDD2022 limpo em COCO e YOLO com splits sem vazamento local."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


CLASSES = ("D00", "D10", "D20", "D40")
CLASS_NAMES = {
    "D00": "Longitudinal crack",
    "D10": "Transverse crack",
    "D20": "Alligator crack",
    "D40": "Pothole",
}
SPLITS = ("train", "val", "test")
RATIOS = np.array((0.80, 0.10, 0.10))


def sequence_number(filename: str) -> int:
    matches = re.findall(r"(\d+)", Path(filename).stem)
    if not matches:
        raise ValueError(f"nome sem identificador numerico: {filename}")
    return int(matches[-1])


def stable_key(seed: int, domain: str, group: str) -> str:
    return hashlib.sha256(f"{seed}:{domain}:{group}".encode()).hexdigest()


def assign_domain_groups(groups: pd.DataFrame, seed: int) -> dict[str, str]:
    """Distribui blocos inteiros buscando balancear imagens, negativos e classes."""
    feature_columns = ["images", "negatives", *CLASSES]
    features = groups[feature_columns].to_numpy(dtype=float)
    totals = features.sum(axis=0)
    targets = RATIOS[:, None] * totals[None, :]
    scales = np.maximum(targets, 1.0)
    feature_weights = np.array((10.0, 2.0, 3.0, 3.0, 3.0, 3.0))
    current = np.zeros_like(targets)

    def objective(values: np.ndarray) -> float:
        score = (feature_weights * np.square((values - targets) / scales)).sum()
        overflow = np.maximum(values - targets * 1.08, 0) / scales
        return float(score + 3.0 * (feature_weights * np.square(overflow)).sum())

    rarity = np.divide(
        features,
        np.maximum(totals, 1.0),
        out=np.zeros_like(features),
        where=np.maximum(totals, 1.0) > 0,
    ).sum(axis=1)
    order = sorted(
        range(len(groups)),
        key=lambda idx: (
            -rarity[idx],
            stable_key(seed, str(groups.iloc[idx]["domain"]), str(groups.iloc[idx]["group_id"])),
        ),
    )

    assignment: dict[str, str] = {}
    assigned_indices = np.full(len(groups), -1, dtype=int)
    for idx in order:
        vector = features[idx]
        scores = []
        for split_idx in range(len(SPLITS)):
            candidate = current.copy()
            candidate[split_idx] += vector
            scores.append(objective(candidate))
        chosen = int(np.argmin(scores))
        current[chosen] += vector
        assigned_indices[idx] = chosen

    # Trocas de blocos refinam o balanceamento sem quebrar o agrupamento temporal.
    domain = str(groups.iloc[0]["domain"])
    domain_seed = int(stable_key(seed, domain, "optimizer")[:8], 16)
    rng = np.random.default_rng(domain_seed)
    current_score = objective(current)
    for _ in range(30_000):
        first, second = rng.integers(0, len(groups), size=2)
        first_split = assigned_indices[first]
        second_split = assigned_indices[second]
        if first == second or first_split == second_split:
            continue
        candidate = current.copy()
        candidate[first_split] += features[second] - features[first]
        candidate[second_split] += features[first] - features[second]
        candidate_score = objective(candidate)
        if candidate_score + 1e-12 < current_score:
            current = candidate
            current_score = candidate_score
            assigned_indices[first], assigned_indices[second] = second_split, first_split

    for idx, split_idx in enumerate(assigned_indices):
        assignment[str(groups.iloc[idx]["group_id"])] = SPLITS[split_idx]
    return assignment


def build_manifest(
    images: pd.DataFrame,
    annotations: pd.DataFrame,
    block_size: int,
    seed: int,
) -> pd.DataFrame:
    labeled = images.loc[images["split"].eq("train")].copy()
    labeled["sequence_number"] = labeled["filename"].map(sequence_number)
    labeled["sequence_block"] = labeled["sequence_number"] // block_size
    labeled["group_id"] = (
        labeled["domain"] + ":" + labeled["sequence_block"].astype(str)
    )

    valid = annotations.loc[
        annotations["is_official"]
        & annotations["bbox_positive"]
        & annotations["bbox_inside_image"]
    ]
    image_classes = (
        valid.groupby(["domain", "filename", "class_id"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CLASSES, fill_value=0)
        .reset_index()
    )
    labeled = labeled.merge(image_classes, on=["domain", "filename"], how="left")
    labeled[list(CLASSES)] = labeled[list(CLASSES)].fillna(0).astype(int)
    labeled["negatives"] = labeled[list(CLASSES)].sum(axis=1).eq(0).astype(int)

    assignments: dict[str, str] = {}
    for domain, domain_images in labeled.groupby("domain", sort=True):
        groups = (
            domain_images.groupby("group_id", as_index=False)
            .agg(
                domain=("domain", "first"),
                images=("filename", "size"),
                negatives=("negatives", "sum"),
                **{class_id: (class_id, "sum") for class_id in CLASSES},
            )
        )
        domain_assignment = assign_domain_groups(groups, seed)
        assignments.update(domain_assignment)

    labeled["clean_split"] = labeled["group_id"].map(assignments)
    labeled["clean_file"] = (
        labeled["domain"] + "/" + labeled["filename"]
    )
    return labeled.sort_values(["clean_split", "domain", "sequence_number"])


def clean_annotations(annotations: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    valid = annotations.loc[
        annotations["is_official"]
        & annotations["bbox_positive"]
        & annotations["bbox_inside_image"]
    ].copy()
    lookup = manifest[["domain", "filename", "clean_split"]]
    return valid.merge(lookup, on=["domain", "filename"], how="inner", validate="many_to_one")


def write_coco(
    split: str,
    manifest: pd.DataFrame,
    annotations: pd.DataFrame,
    output: Path,
) -> None:
    split_images = manifest.loc[manifest["clean_split"].eq(split)].copy()
    split_annotations = annotations.loc[annotations["clean_split"].eq(split)]
    image_ids = {
        (row.domain, row.filename): index
        for index, row in enumerate(split_images.itertuples(index=False), start=1)
    }
    coco_images = [
        {
            "id": image_ids[(row.domain, row.filename)],
            "file_name": f"{row.domain}/{row.filename}",
            "width": int(row.width),
            "height": int(row.height),
            "domain": row.domain,
            "sequence_block": int(row.sequence_block),
        }
        for row in split_images.itertuples(index=False)
    ]
    coco_annotations = []
    for annotation_id, row in enumerate(split_annotations.itertuples(index=False), start=1):
        width = float(row.xmax - row.xmin)
        height = float(row.ymax - row.ymin)
        coco_annotations.append(
            {
                "id": annotation_id,
                "image_id": image_ids[(row.domain, row.filename)],
                "category_id": CLASSES.index(row.class_id) + 1,
                "bbox": [float(row.xmin), float(row.ymin), width, height],
                "area": width * height,
                "iscrowd": 0,
                "truncated": int(row.truncated),
                "difficult": int(row.difficult),
            }
        )
    payload = {
        "info": {
            "description": "RDD2022 cleaned internal split",
            "version": "1.0",
            "split": split,
            "split_policy": "domain-stratified sequence blocks",
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC BY 4.0",
                "url": "https://creativecommons.org/licenses/by/4.0/",
            }
        ],
        "categories": [
            {"id": index + 1, "name": class_id, "supercategory": "road_damage"}
            for index, class_id in enumerate(CLASSES)
        ],
        "images": coco_images,
        "annotations": coco_annotations,
    }
    destination = output / "annotations" / f"instances_{split}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_yolo_labels(
    manifest: pd.DataFrame,
    annotations: pd.DataFrame,
    output: Path,
) -> None:
    labels_root = output / "labels"
    if labels_root.exists():
        shutil.rmtree(labels_root)
    grouped = {
        key: group
        for key, group in annotations.groupby(["clean_split", "domain", "filename"])
    }
    for image in manifest.itertuples(index=False):
        label_path = (
            output
            / "labels"
            / image.clean_split
            / image.domain
            / f"{Path(image.filename).stem}.txt"
        )
        label_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        group = grouped.get((image.clean_split, image.domain, image.filename))
        if group is not None:
            for box in group.itertuples(index=False):
                center_x = ((box.xmin + box.xmax) / 2) / image.width
                center_y = ((box.ymin + box.ymax) / 2) / image.height
                width = (box.xmax - box.xmin) / image.width
                height = (box.ymax - box.ymin) / image.height
                rows.append(
                    f"{CLASSES.index(box.class_id)} {center_x:.8f} {center_y:.8f} "
                    f"{width:.8f} {height:.8f}"
                )
        label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    yaml = [
        f"path: {json.dumps(str(output.resolve()))}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *[f"  {index}: {class_id}" for index, class_id in enumerate(CLASSES)],
    ]
    (output / "dataset.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")


def extract_images(manifest: pd.DataFrame, output: Path) -> None:
    images_root = output / "images"
    expected = {
        Path(image.clean_split) / image.domain / image.filename
        for image in manifest.itertuples(index=False)
    }
    if images_root.exists():
        for existing in images_root.rglob("*"):
            if existing.is_file() and existing.relative_to(images_root) not in expected:
                existing.unlink()
        for directory in sorted(
            (path for path in images_root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass
    for archive_path, archive_images in manifest.groupby("archive", sort=True):
        print(f"Extraindo imagens de {Path(archive_path).name}...")
        with zipfile.ZipFile(archive_path) as zipped:
            for image in archive_images.itertuples(index=False):
                destination = output / "images" / image.clean_split / image.domain / image.filename
                if destination.exists() and destination.stat().st_size == image.file_bytes:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".part")
                with zipped.open(image.archive_member) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                if temporary.stat().st_size != image.file_bytes:
                    raise RuntimeError(f"tamanho incorreto apos extracao: {destination}")
                temporary.replace(destination)


def write_audit(
    manifest: pd.DataFrame,
    annotations: pd.DataFrame,
    challenge: pd.DataFrame,
    block_size: int,
    seed: int,
    output: Path,
) -> None:
    image_summary = (
        manifest.groupby(["clean_split", "domain"])
        .agg(images=("filename", "size"), negatives=("negatives", "sum"))
        .reset_index()
    )
    class_summary = (
        annotations.groupby(["clean_split", "domain", "class_id"])
        .size()
        .rename("instances")
        .reset_index()
    )
    image_summary.to_csv(output / "split_image_summary.csv", index=False)
    class_summary.to_csv(output / "split_class_summary.csv", index=False)

    blocks_per_split = manifest.groupby("group_id")["clean_split"].nunique()
    leakage = int((blocks_per_split > 1).sum())
    split_totals = manifest["clean_split"].value_counts().reindex(SPLITS, fill_value=0)
    class_totals = pd.crosstab(annotations["clean_split"], annotations["class_id"]).reindex(
        index=SPLITS, columns=CLASSES, fill_value=0
    )
    lines = [
        "# Auditoria da preparacao — RDD2022",
        "",
        f"- Semente: `{seed}`.",
        f"- Bloco sequencial: `{block_size}` identificadores consecutivos.",
        f"- Razoes-alvo treino/validacao/teste: `{RATIOS.tolist()}`.",
        f"- Imagens rotuladas limpas: **{len(manifest):,}**.",
        f"- Imagens do teste original sem rotulos, mantidas separadas: **{len(challenge):,}**.",
        f"- Caixas oficiais validas: **{len(annotations):,}**.",
        f"- Blocos presentes em mais de um split: **{leakage}**.",
        "",
        "## Totais por split",
        "",
        "| Split | Imagens | D00 | D10 | D20 | D40 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        values = class_totals.loc[split]
        lines.append(
            f"| {split} | {split_totals[split]:,} | "
            + " | ".join(f"{int(values[c]):,}" for c in CLASSES)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Regras aplicadas",
            "",
            "1. Apenas `D00`, `D10`, `D20` e `D40` sao alvos.",
            "2. Classes adicionais sao removidas das labels; imagens que ficam sem alvo sao preservadas como negativas.",
            "3. Caixas sem area positiva ou fora da imagem sao descartadas.",
            "4. Todos os quadros de um mesmo bloco sequencial ficam no mesmo split.",
            "5. A distribuicao e feita separadamente por dominio, equilibrando imagens, negativos e instancias por classe.",
            "6. O teste original do desafio permanece intocado e nao participa da avaliacao interna.",
        ]
    )
    (output / "AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_lodo_manifests(manifest: pd.DataFrame, output: Path) -> None:
    destination = output / "lodo_manifests"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for heldout_domain in sorted(manifest["domain"].unique()):
        lodo = manifest.copy()
        lodo["lodo_split"] = np.where(
            lodo["domain"].eq(heldout_domain),
            "test",
            np.where(lodo["clean_split"].eq("val"), "val", "train"),
        )
        lodo["heldout_domain"] = heldout_domain
        lodo.to_csv(destination / f"holdout_{heldout_domain}.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eda", type=Path, default=Path("outputs/eda_rdd2022"))
    parser.add_argument("--output", type=Path, default=Path("data/rdd2022/clean"))
    parser.add_argument("--block-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--extract-images",
        action="store_true",
        help="extrai as imagens rotuladas para os diretorios train/val/test",
    )
    args = parser.parse_args()
    if args.block_size < 1:
        parser.error("--block-size deve ser positivo")
    args.output.mkdir(parents=True, exist_ok=True)

    images = pd.read_csv(args.eda / "images.csv")
    annotations = pd.read_csv(args.eda / "annotations.csv")
    manifest = build_manifest(images, annotations, args.block_size, args.seed)
    clean = clean_annotations(annotations, manifest)
    challenge = images.loc[images["split"].eq("test")].copy()

    manifest.to_csv(args.output / "split_manifest.csv", index=False)
    challenge.to_csv(args.output / "challenge_test_manifest.csv", index=False)
    clean.to_csv(args.output / "clean_annotations.csv", index=False)
    for split in SPLITS:
        write_coco(split, manifest, clean, args.output)
    write_yolo_labels(manifest, clean, args.output)
    write_lodo_manifests(manifest, args.output)
    write_audit(manifest, clean, challenge, args.block_size, args.seed, args.output)
    if args.extract_images:
        extract_images(manifest, args.output)
    print(f"Dataset limpo preparado em {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
