# Compute-Constrained Road Damage Detection

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

The paper is written in English using the official IEEEtran conference class.
Build it with:

```bash
cd paper
make
```

## Reproduce the data pipeline

The dataset is not redistributed by this repository. Download RDD2022 from its
[Figshare record](https://figshare.com/ndownloader/articles/21431547/versions/1),
then run the documented Python pipeline:

```bash
python download_rdd2022.py
python prepare_rdd2022.py
python validate_rdd2022.py
python eda_rdd2022.py
```

Experiment entry points and the frozen CPU protocol are documented in
[`experiments/README.md`](experiments/README.md) and
[`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md).

## Repository contents

- `artifacts/`: compact, machine-readable frozen metrics and training history;
- `experiments/`: training, evaluation, diagnostics, and experiment log;
- `paper/`: IEEE manuscript, bibliography, template provenance, and figure;
- root Python scripts: download, cleaning, validation, and EDA.

Datasets, dependency environments, checkpoints, raw predictions, and generated
experiment directories are excluded from Git history. Artifact SHA-256 values
are recorded so externally hosted weights and predictions can be verified.

## Author

Mateus Gomes de Araújo, Computer Engineering, University of Brasília (UnB).
Contact: `mathweuzz@gmail.com`.
