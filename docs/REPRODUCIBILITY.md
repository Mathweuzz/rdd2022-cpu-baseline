# Reproducibility map

This document maps each public command to its inputs and outputs. Large or
licensed artifacts are intentionally outside Git history.

## Data flow

| Stage | Command | Main input | Main output |
|---|---|---|---|
| Download | `make download` | Figshare API | `data/rdd2022/raw/`, `data/rdd2022/archives/RDD2022/` |
| EDA | `make eda` | Seven domain ZIP files | `outputs/eda_rdd2022/` |
| Clean/split | `make prepare` | EDA image and annotation tables | `data/rdd2022/clean/` |
| Audit | `make validate` | Clean COCO/YOLO dataset | Validation report on stdout |
| CPU protocol | `make protocol` | Clean split manifest | `outputs/cpu_protocol_2800/` |
| Training | `make train` | Protocol and COCO-pretrained YOLO11n | `outputs/yolo11n_joint_cpu_final_seed2026/` |
| Evaluation | command in README | Final-epoch checkpoint and internal test | Prediction and evaluation JSON files |
| Paper | `make paper` | Tracked source, figure, metrics | `paper/main.pdf` |

The download is approximately 13 GB. Extracting images and retaining generated
artifacts requires substantially more disk space. Training took 28.5 minutes on
the reported Intel Core i5-10500T; timings will differ by host.

## Frozen scientific controls

- Seed 2026 and deterministic mode.
- Sequence blocks never cross train, validation, and internal-test partitions.
- Exactly 400 optimization images from each of seven domains.
- Domain-specific negative prevalence is retained.
- Final epoch selected without internal-test feedback.
- One shared prediction set is sliced for pooled, class, scale, domain, and
  negative-image analysis.
- No repeated-seed uncertainty is claimed.

`artifacts/frozen_protocol.json`, `artifacts/training_config.yaml`,
`artifacts/training_history.csv`, and `artifacts/test_evaluation.json` are the
compact public record. The GitHub release contains the checkpoint and full
prediction file; `paper/RESULTS_PROVENANCE.md` contains their SHA-256 hashes.

## Expected headline results

- Test images / instances: 3,703 / 5,486.
- COCO mAP / AP50 / AP75: 3.26% / 8.51% / 1.82%.
- Domain-macro / worst-domain mAP: 3.47% / 0.42%.
- Negative false-positive rate at score 0.25: 0.92%.
- Training / complete evaluation time: 1,709.6 s / 162.1 s.

Exact floating-point values are stored in the JSON artifact. Small timing
differences are expected across hardware; metric differences require an issue
using the reproducibility template.
