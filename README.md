# Compute-Constrained Road Damage Detection

[![CI](https://github.com/Mathweuzz/rdd2022-cpu-baseline/actions/workflows/ci.yml/badge.svg)](https://github.com/Mathweuzz/rdd2022-cpu-baseline/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22285375.svg)](https://doi.org/10.5281/zenodo.22285375)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/Mathweuzz/rdd2022-cpu-baseline)](https://github.com/Mathweuzz/rdd2022-cpu-baseline/releases/latest)
[![Paper](https://img.shields.io/badge/paper-IEEEtran-b31b1b.svg)](paper/main.pdf)

Reproducible CPU-only object detection on the multinational RDD2022 benchmark,
with sequence-aware splitting, explicit negative-image preservation, equal-domain
training, and domain-sliced COCO evaluation.

The frozen YOLO11n run uses 2,800 training images (400 per acquisition domain),
10 epochs, 320-by-320 inputs, and no GPU. On the held-out 3,703-image internal
test split it obtains 3.26% COCO mAP, 8.51% AP50, and 43.4 images/s. The result
is intended as an auditable lower-compute reference, not a state-of-the-art
claim.

## Paper

- [Compiled IEEE paper](paper/main.pdf)
- [LaTeX source](paper/main.tex)
- [Results provenance](paper/RESULTS_PROVENANCE.md)
- [Permanent Zenodo archive](https://doi.org/10.5281/zenodo.22285376)

The paper is written in English using the official IEEEtran conference class.
Build it with:

```bash
cd paper
make
```

## Installation

The published run used Python 3.14, while CI verifies the lightweight pipeline
on Python 3.12. A virtual environment is strongly recommended:

```bash
git clone https://github.com/Mathweuzz/rdd2022-cpu-baseline.git
cd rdd2022-cpu-baseline
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[experiments]"
```

Install manifests pin patched compatible versions. The exact historical versions
used to produce the published metrics remain recorded in
`paper/RESULTS_PROVENANCE.md`; they are provenance, not a recommendation to
install packages with known advisories.

## Reproduce the complete pipeline

The dataset is not redistributed by this repository. Download RDD2022 from its
[Figshare record](https://figshare.com/ndownloader/articles/21431547/versions/1),
then run:

```bash
make download   # approximately 13 GB, with Figshare MD5 verification
make eda        # reads the seven domain ZIP files directly
make prepare    # sequence-blocked split plus COCO/YOLO export
make validate   # structural, geometry, and leakage checks
make protocol   # frozen 2,800-image equal-domain training lists
```

The commands intentionally remain separate so every generated artifact can be
audited before the next stage. Run `make help` for all entry points.

## Reproduce training and evaluation

The following command implements the exact frozen CPU defaults from the paper:

```bash
make train
```

After training, evaluate the final-epoch checkpoint with the low-memory path:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python experiments/evaluate_yolo_checkpoint.py \
  --model outputs/yolo11n_joint_cpu_final_seed2026/weights/last.pt \
  --split test \
  --output outputs/yolo11n_joint_cpu_final_seed2026/test_evaluation \
  --imgsz 320 --batch 1 --workers 0 --threads 1
```

To try the released checkpoint on one image or a directory:

```bash
gh release download v0.1.0 --pattern 'yolo11n-rdd2022-cpu-seed2026.*' --dir models
python experiments/predict.py \
  --model models/yolo11n-rdd2022-cpu-seed2026.safetensors \
  --config models/yolo11n-rdd2022-cpu-seed2026.yaml \
  --source path/to/image-or-directory
```

Experiment entry points and the frozen CPU protocol are documented in
[`experiments/README.md`](experiments/README.md) and
[`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md). The end-to-end artifact map
is in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Repository contents

- `artifacts/`: compact, machine-readable frozen metrics and training history;
- `experiments/`: training, evaluation, diagnostics, and experiment log;
- `paper/`: IEEE manuscript, bibliography, template provenance, and figure;
- `tests/`: lightweight integrity and determinism checks;
- root Python scripts: download, cleaning, validation, and EDA.

Datasets, dependency environments, checkpoints, raw predictions, and generated
experiment directories are excluded from Git history. Artifact SHA-256 values
are recorded so externally hosted weights and predictions can be verified.

## Verification

```bash
make test
make paper
```

GitHub Actions repeats the lightweight tests and builds the paper from a clean
checkout. The full 28.5-minute training run is intentionally not repeated in CI.

## Author

Mateus Gomes de Araújo, Computer Engineering, University of Brasília (UnB).
Contact: `mathweuzz@gmail.com`.

## Citation

The versioned software archive is available under DOI
[`10.5281/zenodo.22285376`](https://doi.org/10.5281/zenodo.22285376). Use the
concept DOI [`10.5281/zenodo.22285375`](https://doi.org/10.5281/zenodo.22285375)
to cite the project independently of a specific release. Machine-readable
citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

Original software in this repository is released under the [MIT License](LICENSE).
The manuscript and original figures remain copyright © 2026 Mateus Gomes de
Araújo. The vendored IEEEtran files retain their upstream license and copyright
notices. Ultralytics and the derived checkpoint are subject to AGPL-3.0 upstream
terms. RDD2022 is not redistributed and remains subject to its source terms.
See [third-party notices](THIRD_PARTY_NOTICES.md) for scope details.
