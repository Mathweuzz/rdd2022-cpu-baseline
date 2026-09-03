#!/usr/bin/env python3
"""Valida integridade estrutural dos artefatos COCO/YOLO do RDD2022."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SPLITS = ("train", "val", "test")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/rdd2022/clean"))
    args = parser.parse_args()
    root = args.dataset
    manifest = pd.read_csv(root / "split_manifest.csv")
    annotations = pd.read_csv(root / "clean_annotations.csv")
    failures: list[str] = []

    if manifest.duplicated(["domain", "filename"]).any():
        failures.append("chaves de imagem duplicadas no manifest")
    leakage = manifest.groupby("group_id")["clean_split"].nunique()
    if (leakage > 1).any():
        failures.append("blocos sequenciais presentes em mais de um split")
    if not annotations["class_id"].isin(("D00", "D10", "D20", "D40")).all():
        failures.append("classes nao oficiais nas anotacoes limpas")
    if not (annotations["bbox_positive"] & annotations["bbox_inside_image"]).all():
        failures.append("caixas invalidas nas anotacoes limpas")

    expected_images = {
        Path(row.clean_split) / row.domain / row.filename
        for row in manifest.itertuples(index=False)
    }
    expected_labels = {
        Path(row.clean_split) / row.domain / f"{Path(row.filename).stem}.txt"
        for row in manifest.itertuples(index=False)
    }
    actual_images = {
        path.relative_to(root / "images")
        for path in (root / "images").rglob("*")
        if path.is_file()
    }
    actual_labels = {
        path.relative_to(root / "labels")
        for path in (root / "labels").rglob("*")
        if path.is_file()
    }
    if actual_images != expected_images:
        failures.append(
            f"arquivos de imagem fisicos diferem do manifest "
            f"(faltam {len(expected_images - actual_images)}, sobram {len(actual_images - expected_images)})"
        )
    if actual_labels != expected_labels:
        failures.append(
            f"arquivos de label fisicos diferem do manifest "
            f"(faltam {len(expected_labels - actual_labels)}, sobram {len(actual_labels - expected_labels)})"
        )

    total_yolo_boxes = 0
    for split in SPLITS:
        split_manifest = manifest[manifest["clean_split"].eq(split)]
        split_annotations = annotations[annotations["clean_split"].eq(split)]
        coco = json.loads((root / "annotations" / f"instances_{split}.json").read_text())
        coco_images = coco["images"]
        coco_annotations = coco["annotations"]
        if len(coco_images) != len(split_manifest):
            failures.append(f"contagem COCO de imagens incorreta em {split}")
        if len(coco_annotations) != len(split_annotations):
            failures.append(f"contagem COCO de anotacoes incorreta em {split}")
        image_ids = {image["id"] for image in coco_images}
        if len(image_ids) != len(coco_images):
            failures.append(f"IDs COCO de imagem duplicados em {split}")
        if any(annotation["image_id"] not in image_ids for annotation in coco_annotations):
            failures.append(f"referencia COCO de imagem invalida em {split}")

        for row in split_manifest.itertuples(index=False):
            image_path = root / "images" / split / row.domain / row.filename
            label_path = root / "labels" / split / row.domain / f"{Path(row.filename).stem}.txt"
            if not image_path.is_file() or image_path.stat().st_size != row.file_bytes:
                failures.append(f"imagem ausente ou com tamanho incorreto: {image_path}")
            if not label_path.is_file():
                failures.append(f"label YOLO ausente: {label_path}")
                continue
            for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
                fields = line.split()
                if len(fields) != 5:
                    failures.append(f"label YOLO malformada: {label_path}:{line_number}")
                    continue
                try:
                    class_id = int(fields[0])
                    coordinates = [float(value) for value in fields[1:]]
                except ValueError:
                    failures.append(f"valor YOLO invalido: {label_path}:{line_number}")
                    continue
                if class_id not in range(4):
                    failures.append(f"classe YOLO invalida: {label_path}:{line_number}")
                if not all(0 <= value <= 1 for value in coordinates):
                    failures.append(f"coordenada YOLO fora de [0,1]: {label_path}:{line_number}")
                if coordinates[2] <= 0 or coordinates[3] <= 0:
                    failures.append(f"caixa YOLO sem area: {label_path}:{line_number}")
                total_yolo_boxes += 1

    if total_yolo_boxes != len(annotations):
        failures.append(
            f"numero de caixas YOLO ({total_yolo_boxes}) difere da tabela ({len(annotations)})"
        )

    lodo_files = sorted((root / "lodo_manifests").glob("holdout_*.csv"))
    if len(lodo_files) != manifest["domain"].nunique():
        failures.append("quantidade incorreta de manifests leave-one-domain-out")
    for path in lodo_files:
        lodo = pd.read_csv(path)
        heldout = lodo["heldout_domain"].iloc[0]
        if not lodo.loc[lodo["domain"].eq(heldout), "lodo_split"].eq("test").all():
            failures.append(f"holdout incorreto em {path.name}")
        if lodo.loc[~lodo["domain"].eq(heldout), "lodo_split"].eq("test").any():
            failures.append(f"dominio nao holdout no teste de {path.name}")

    report = [
        "# Validacao do dataset preparado",
        "",
        f"- Imagens verificadas: **{len(manifest):,}**.",
        f"- Caixas COCO/YOLO verificadas: **{total_yolo_boxes:,}**.",
        f"- Manifests leave-one-domain-out: **{len(lodo_files)}**.",
        f"- Falhas: **{len(failures)}**.",
    ]
    if failures:
        report.extend(["", "## Falhas", ""] + [f"- {failure}" for failure in failures[:100]])
    (root / "VALIDATION.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
