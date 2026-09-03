#!/usr/bin/env python3
"""EDA reproduzivel do RDD2022 diretamente nos ZIPs por dominio."""

from __future__ import annotations

import argparse
import io
import math
import os
import xml.etree.ElementTree as ET
import zipfile
from contextlib import ExitStack
from pathlib import Path, PurePosixPath

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rdd2022-matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps


OFFICIAL_CLASSES = ("D00", "D10", "D20", "D40")
CLASS_NAMES = {
    "D00": "fissura longitudinal",
    "D10": "fissura transversal",
    "D20": "fissura em malha",
    "D40": "buraco",
}
CLASS_NAMES_EN = {
    "D00": "longitudinal crack",
    "D10": "transverse crack",
    "D20": "alligator crack",
    "D40": "pothole",
}
CLASS_COLORS = {
    "D00": "#1f77b4",
    "D10": "#ff7f0e",
    "D20": "#2ca02c",
    "D40": "#d62728",
}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def number(node: ET.Element | None, path: str) -> float:
    text = node.findtext(path) if node is not None else None
    try:
        return float(text) if text not in (None, "") else math.nan
    except ValueError:
        return math.nan


def flag(node: ET.Element, path: str) -> int:
    value = number(node, path)
    return int(value) if math.isfinite(value) else 0


def split_from_member(member: str) -> str:
    parts = PurePosixPath(member).parts
    return "test" if "test" in parts else "train" if "train" in parts else "unknown"


def scan_archive(archive: Path) -> tuple[list[dict], list[dict], list[str]]:
    domain = archive.stem
    image_rows: dict[tuple[str, str], dict] = {}
    annotation_rows: list[dict] = []
    errors: list[str] = []

    with zipfile.ZipFile(archive) as zipped:
        image_members = [
            info
            for info in zipped.infolist()
            if not info.is_dir() and info.filename.lower().endswith(IMAGE_SUFFIXES)
        ]
        for info in image_members:
            split = split_from_member(info.filename)
            filename = PurePosixPath(info.filename).name
            image_rows[(split, filename)] = {
                "domain": domain,
                "split": split,
                "filename": filename,
                "archive": str(archive),
                "archive_member": info.filename,
                "compressed_bytes": info.compress_size,
                "file_bytes": info.file_size,
                "width": math.nan,
                "height": math.nan,
                "depth": math.nan,
                "has_xml": False,
                "objects_all": 0,
                "objects_official": 0,
            }

        xml_members = [
            info
            for info in zipped.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".xml")
        ]
        for info in xml_members:
            try:
                root = ET.fromstring(zipped.read(info))
            except ET.ParseError as error:
                errors.append(f"{domain}: XML invalido {info.filename}: {error}")
                continue

            split = split_from_member(info.filename)
            filename = root.findtext("filename") or f"{PurePosixPath(info.filename).stem}.jpg"
            width = number(root, "size/width")
            height = number(root, "size/height")
            depth = number(root, "size/depth")
            key = (split, PurePosixPath(filename).name)
            image = image_rows.get(key)
            if image is None:
                errors.append(f"{domain}: XML sem imagem correspondente: {info.filename}")
                continue

            objects = root.findall("object")
            official_count = sum(
                (obj.findtext("name") or "").strip() in OFFICIAL_CLASSES
                for obj in objects
            )
            image.update(
                width=width,
                height=height,
                depth=depth,
                has_xml=True,
                objects_all=len(objects),
                objects_official=official_count,
            )

            for obj in objects:
                class_id = (obj.findtext("name") or "UNKNOWN").strip()
                xmin = number(obj, "bndbox/xmin")
                ymin = number(obj, "bndbox/ymin")
                xmax = number(obj, "bndbox/xmax")
                ymax = number(obj, "bndbox/ymax")
                bbox_width = xmax - xmin
                bbox_height = ymax - ymin
                finite = all(math.isfinite(v) for v in (xmin, ymin, xmax, ymax))
                positive = finite and bbox_width > 0 and bbox_height > 0
                inside = (
                    positive
                    and math.isfinite(width)
                    and math.isfinite(height)
                    and xmin >= 0
                    and ymin >= 0
                    and xmax <= width
                    and ymax <= height
                )
                annotation_rows.append(
                    {
                        "domain": domain,
                        "split": split,
                        "filename": key[1],
                        "class_id": class_id,
                        "class_name": CLASS_NAMES.get(class_id, "classe nao oficial"),
                        "is_official": class_id in OFFICIAL_CLASSES,
                        "image_width": width,
                        "image_height": height,
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "bbox_width": bbox_width,
                        "bbox_height": bbox_height,
                        "bbox_area": bbox_width * bbox_height if positive else math.nan,
                        "relative_width": bbox_width / width if positive and width else math.nan,
                        "relative_height": bbox_height / height if positive and height else math.nan,
                        "relative_area": (
                            bbox_width * bbox_height / (width * height)
                            if positive and width and height
                            else math.nan
                        ),
                        "aspect_ratio": bbox_width / bbox_height if positive else math.nan,
                        "truncated": flag(obj, "truncated"),
                        "difficult": flag(obj, "difficult"),
                        "occluded": flag(obj, "occluded"),
                        "bbox_positive": positive,
                        "bbox_inside_image": inside,
                    }
                )

    return list(image_rows.values()), annotation_rows, errors


def save_overview(images: pd.DataFrame, annotations: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))

    counts = images.groupby(["domain", "split"]).size().unstack(fill_value=0)
    counts = counts.sort_values("train", ascending=False)
    counts.plot(kind="bar", stacked=True, ax=axes[0, 0], color=["#ef8a62", "#67a9cf"])
    axes[0, 0].set(title="Images by domain and split", xlabel="", ylabel="images")
    axes[0, 0].tick_params(axis="x", rotation=35)
    axes[0, 0].legend(title="split")

    official = annotations[annotations["is_official"]]
    class_counts = official["class_id"].value_counts().reindex(OFFICIAL_CLASSES)
    axes[0, 1].bar(
        class_counts.index,
        class_counts.values,
        color=[CLASS_COLORS[c] for c in class_counts.index],
    )
    axes[0, 1].set(
        title="Instances by official class", xlabel="class", ylabel="bounding boxes"
    )
    for index, value in enumerate(class_counts.values):
        axes[0, 1].text(index, value, f"{value:,}", ha="center", va="bottom")

    matrix = pd.crosstab(official["domain"], official["class_id"]).reindex(columns=OFFICIAL_CLASSES, fill_value=0)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="g",
        cmap="Blues",
        ax=axes[1, 0],
        cbar_kws={"label": "bounding boxes"},
    )
    axes[1, 0].set(title="Classes by domain", xlabel="class", ylabel="")

    positive = official[official["bbox_positive"] & (official["relative_area"] > 0)].copy()
    sns.boxplot(
        data=positive,
        x="class_id",
        y="relative_area",
        order=OFFICIAL_CLASSES,
        hue="class_id",
        palette=CLASS_COLORS,
        legend=False,
        showfliers=False,
        ax=axes[1, 1],
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(
        title="Relative bounding-box area (log scale)",
        xlabel="class",
        ylabel="bounding-box area / image area",
    )

    fig.suptitle("RDD2022 — overview", fontsize=24, y=1.01)
    fig.tight_layout()
    fig.savefig(output / "overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_sample_grid(images: pd.DataFrame, annotations: pd.DataFrame, output: Path) -> None:
    candidates = annotations[
        annotations["is_official"]
        & annotations["bbox_positive"]
        & annotations["bbox_inside_image"]
    ].copy()
    chosen = []
    for class_id in OFFICIAL_CLASSES:
        group = candidates[candidates["class_id"] == class_id].drop_duplicates(
            ["domain", "filename"]
        )
        if len(group):
            chosen.append(group.sample(min(3, len(group)), random_state=42))
    selected = pd.concat(chosen, ignore_index=True).drop_duplicates(["domain", "filename"])
    selected = selected.head(12)
    image_lookup = images.set_index(["domain", "filename"])

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flat
    with ExitStack() as stack:
        archives: dict[str, zipfile.ZipFile] = {}
        for axis, sample in zip(axes, selected.itertuples(index=False), strict=False):
            image_row = image_lookup.loc[(sample.domain, sample.filename)]
            if isinstance(image_row, pd.DataFrame):
                image_row = image_row.iloc[0]
            archive_path = str(image_row["archive"])
            if archive_path not in archives:
                archives[archive_path] = stack.enter_context(zipfile.ZipFile(archive_path))
            payload = archives[archive_path].read(image_row["archive_member"])
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(payload))).convert("RGB")
            axis.imshow(image)

            boxes = annotations[
                (annotations["domain"] == sample.domain)
                & (annotations["filename"] == sample.filename)
                & annotations["is_official"]
                & annotations["bbox_positive"]
            ]
            for box in boxes.itertuples(index=False):
                axis.add_patch(
                    Rectangle(
                        (box.xmin, box.ymin),
                        box.bbox_width,
                        box.bbox_height,
                        fill=False,
                        linewidth=2,
                        edgecolor=CLASS_COLORS[box.class_id],
                    )
                )
                axis.text(
                    box.xmin,
                    max(0, box.ymin - 3),
                    box.class_id,
                    color="white",
                    fontsize=9,
                    bbox={"facecolor": CLASS_COLORS[box.class_id], "alpha": 0.8, "pad": 1},
                )
            axis.set_title(f"{sample.domain}\n{sample.filename}", fontsize=11)
            axis.axis("off")

    for axis in list(axes)[len(selected) :]:
        axis.axis("off")
    handles = [
        Rectangle(
            (0, 0), 1, 1, color=CLASS_COLORS[c], label=f"{c}: {CLASS_NAMES_EN[c]}"
        )
        for c in OFFICIAL_CLASSES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=11)
    fig.suptitle("Annotated RDD2022 samples", fontsize=22)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(output / "sample_bboxes.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_report(
    images: pd.DataFrame,
    annotations: pd.DataFrame,
    errors: list[str],
    output: Path,
) -> None:
    train = images[images["split"] == "train"]
    official = annotations[annotations["is_official"]]
    unknown = annotations[~annotations["is_official"]]
    negatives = train[train["objects_official"] == 0]
    invalid = official[~official["bbox_positive"]]
    outside = official[official["bbox_positive"] & ~official["bbox_inside_image"]]
    class_counts = official["class_id"].value_counts().reindex(OFFICIAL_CLASSES, fill_value=0)
    dominant_domain = images["domain"].value_counts().idxmax()
    dominant_count = int(images["domain"].value_counts().max())
    resolutions = (
        train.dropna(subset=["width", "height"])
        .assign(resolution=lambda x: x["width"].astype(int).astype(str) + "x" + x["height"].astype(int).astype(str))
        ["resolution"]
        .value_counts()
        .head(8)
    )
    unknown_counts = unknown["class_id"].value_counts()
    negative_by_domain = train.groupby("domain").agg(
        train_images=("filename", "size"),
        negative_images=("objects_official", lambda values: (values == 0).sum()),
    )
    negative_by_domain["negative_rate"] = (
        negative_by_domain["negative_images"] / negative_by_domain["train_images"]
    )
    bbox_area = official.loc[official["bbox_positive"], "relative_area"]
    small_box_rate = (bbox_area < 0.01).mean()

    lines = [
        "# EDA inicial — RDD2022",
        "",
        "## Resumo executivo",
        "",
        f"- **{len(images):,} imagens**: {(images['split'] == 'train').sum():,} de treino e {(images['split'] == 'test').sum():,} de teste.",
        f"- **{len(official):,} caixas oficiais** nas quatro classes do desafio.",
        f"- **{len(negatives):,} imagens de treino sem dano oficial** ({len(negatives) / len(train):.1%} do treino).",
        f"- **{len(unknown):,} objetos de classes nao oficiais**; distribuicao: {unknown_counts.to_dict()}.",
        f"- **{len(invalid):,} caixas sem area positiva** e **{len(outside):,} caixas positivas fora dos limites**.",
        f"- O maior dominio e **{dominant_domain}**, com {dominant_count:,} imagens ({dominant_count / len(images):.1%} do total).",
        f"- **{small_box_rate:.1%} das caixas oficiais ocupam menos de 1% da imagem**, indicando forte presenca de objetos pequenos.",
        "",
        "## Classes oficiais",
        "",
        "| Classe | Significado | Instancias | Proporcao |",
        "|---|---|---:|---:|",
    ]
    for class_id, count in class_counts.items():
        lines.append(
            f"| {class_id} | {CLASS_NAMES[class_id]} | {count:,} | {count / len(official):.1%} |"
        )
    nonzero = class_counts[class_counts > 0]
    lines.extend(
        [
            "",
            f"Razao entre a classe mais e menos frequente: **{nonzero.max() / nonzero.min():.2f}x**.",
            "",
            "## Resolucao das imagens de treino",
            "",
            "| Resolucao | Imagens |",
            "|---|---:|",
        ]
    )
    for resolution, count in resolutions.items():
        lines.append(f"| {resolution} | {count:,} |")
    lines.extend(
        [
            "",
            "## Imagens negativas por dominio",
            "",
            "| Dominio | Treino | Sem classe oficial | Proporcao |",
            "|---|---:|---:|---:|",
        ]
    )
    for domain, row in negative_by_domain.iterrows():
        lines.append(
            f"| {domain} | {int(row['train_images']):,} | "
            f"{int(row['negative_images']):,} | {row['negative_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Implicacoes para modelagem",
            "",
            "1. Fazer a divisao de validacao por dominio/pais; uma divisao puramente aleatoria pode superestimar a generalizacao.",
            "2. Decidir explicitamente o tratamento de classes nao oficiais (por exemplo, `Repair`) antes da conversao para YOLO/COCO.",
            "3. Preservar imagens negativas no treino para reduzir falsos positivos, mas controlar sua proporcao.",
            "4. Usar treinamento multi-escala ou tiling: as resolucoes e os tamanhos relativos das caixas variam bastante entre dominios.",
            "5. Corrigir ou excluir caixas invalidas/fora dos limites durante a preparacao, registrando cada alteracao.",
            "",
            "## Artefatos",
            "",
            "- `images.csv`: inventario por imagem.",
            "- `annotations.csv`: uma linha por objeto anotado.",
            "- `domain_summary.csv` e `class_summary.csv`: agregacoes principais.",
            "- `overview.png`: composicao, classes, dominios e escala das caixas.",
            "- `sample_bboxes.png`: auditoria visual de amostras anotadas.",
        ]
    )
    if errors:
        lines.extend(["", "## Alertas de leitura", ""] + [f"- {error}" for error in errors])

    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archives",
        type=Path,
        default=Path("data/rdd2022/archives/RDD2022"),
        help="diretorio que contem os ZIPs por dominio",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/eda_rdd2022"),
        help="diretorio dos resultados",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    archives = sorted(args.archives.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"nenhum ZIP encontrado em {args.archives}")

    all_images: list[dict] = []
    all_annotations: list[dict] = []
    all_errors: list[str] = []
    for archive in archives:
        print(f"Lendo {archive.name}...")
        images, annotations, errors = scan_archive(archive)
        all_images.extend(images)
        all_annotations.extend(annotations)
        all_errors.extend(errors)

    images_df = pd.DataFrame(all_images).sort_values(["domain", "split", "filename"])
    annotations_df = pd.DataFrame(all_annotations).sort_values(
        ["domain", "filename", "class_id"]
    )
    images_df.to_csv(args.output / "images.csv", index=False)
    annotations_df.to_csv(args.output / "annotations.csv", index=False)

    domain_summary = (
        images_df.groupby(["domain", "split"])
        .agg(images=("filename", "size"), bytes=("file_bytes", "sum"))
        .reset_index()
    )
    domain_summary.to_csv(args.output / "domain_summary.csv", index=False)
    class_summary = (
        annotations_df.groupby(["domain", "class_id", "is_official"])
        .agg(
            instances=("filename", "size"),
            images=("filename", "nunique"),
            median_relative_area=("relative_area", "median"),
        )
        .reset_index()
    )
    class_summary.to_csv(args.output / "class_summary.csv", index=False)

    save_overview(images_df, annotations_df, args.output)
    save_sample_grid(images_df, annotations_df, args.output)
    write_report(images_df, annotations_df, all_errors, args.output)
    print(f"EDA concluido: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
