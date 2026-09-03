# Model card: RDD2022 YOLO11n CPU baseline

## Model details

- Architecture: YOLO11n, 2,590,620 parameters.
- Initialization: official COCO-pretrained `yolo11n.pt`.
- Task: four-class road-damage object detection (D00, D10, D20, D40).
- Input: RGB images resized to 320 by 320 pixels.
- Training: 2,800 images, 400 per acquisition domain, 10 epochs, CPU only.
- Checkpoint rule: final epoch; no internal-test selection.

## Intended use

The checkpoint is a transparent low-compute research baseline for reproducing
the accompanying paper and studying domain-, class-, scale-, and negative-image
behavior. It may support education and pipeline development.

It is **not suitable for autonomous maintenance decisions, safety-critical road
assessment, or production deployment** without substantially stronger local
validation. Human review is required for any operational use.

## Evaluation

On the sequence-blocked 3,703-image internal test split:

| Metric | Value |
|---|---:|
| COCO mAP (0.50:0.95) | 3.26% |
| AP50 / AP75 | 8.51% / 1.82% |
| Small / medium / large AP | 0.26% / 2.71% / 3.63% |
| Domain-macro / worst-domain mAP | 3.47% / 0.42% |
| Negative-image false-positive rate at score 0.25 | 0.92% |
| Measured model-stage throughput | 43.4 images/s |

Full metrics are in `artifacts/test_evaluation.json`.

## Limitations and biases

- Absolute accuracy is low and does not approach unconstrained RDD2022 systems.
- All seven evaluated domains occur in training; results do not establish
  generalization to an unseen geography or sensor.
- Geographic domain, sensor, resolution, annotation practice, weather, and road
  appearance are confounded.
- Only one deterministic seed was trained.
- Equal image quotas retain unequal positive supervision because domain-specific
  negative prevalence is preserved.
- The 320-pixel input is particularly restrictive for small cracks.

## Reproducibility and license

Configuration, metrics, hashes, hardware, and software versions are documented
in `paper/RESULTS_PROVENANCE.md`. The training code written for this repository
is MIT-licensed. Ultralytics software and the released derived checkpoint are
subject to the upstream AGPL-3.0 licensing terms; commercial users should review
the upstream licensing options. RDD2022 is not redistributed.
