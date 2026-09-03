# Experiment log

## M1a — CPU integration smoke test

- Date: 2026-09-02
- Status: passed
- Purpose: validate data loading, target conversion, optimization, inference, and
  checkpoint serialization. These values are **not scientific results**.
- Model: TorchVision SSDLite320 with MobileNetV3-Large backbone.
- Dataset path: cleaned RDD2022 COCO artifacts.
- Available training images: 31,021.
- Test subset: 8 training images, one optimization batch, one validation image.
- Seed: 2026.
- Runtime: CPU, 8 threads; CUDA unavailable.
- Software: Python 3.14.7, PyTorch 2.10.0+cu128, TorchVision 0.25.0+cu128.

Two paths were validated:

1. Random initialization (`outputs/baseline_smoke/`).
2. ImageNet-pretrained MobileNetV3 backbone from the official PyTorch model
   repository (`outputs/baseline_smoke_pretrained/`).

Both runs produced finite losses, completed inference, and saved reloadable
checkpoints. The pretrained run processed one optimization batch in 0.26 s and
one validation inference in 0.03 s on this CPU. These timings are only pipeline
diagnostics and must not appear in comparison tables.

COCO evaluation was subsequently enabled with `pycocotools==2.0.11` and
validated on eight images. Metrics from the one-step random detection head are
diagnostic only and are intentionally excluded from the manuscript.

## M1b — Small-subset overfit diagnostic

- Status: passed with an architectural caveat.
- Data: eight fixed images, no augmentation.
- Optimization: 40 batches over 10 passes through the subset.
- Median nonzero total loss: 16.93 over the first five nonzero batches and 7.00
  over the last five.
- Median classification loss fell from 9.90 to 4.22 (57.4%).
- Output: `outputs/baseline_overfit_test/`.

Three batches containing only negative images produced zero loss. TorchVision's
SSDLite hard-negative mining selects negatives in proportion to matched positive
anchors, so an all-negative batch does not contribute a training signal. This is
an important limitation for RDD2022, where negative-image prevalence is one of
the planned ablations. SSDLite therefore remains an integration baseline, not
the preferred scientific baseline for studying negative sampling.

Official implementation references:

- <https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html>
- <https://docs.pytorch.org/vision/master/models/ssdlite.html>

## Remaining work before M1 is frozen

## M1c — Primary convolutional baseline qualification

RetinaNet-ResNet50-FPN-v2 was tested against SSDLite on the same batch of two
negative Czech images:

| Model | Classification loss | Gradient L1 | Learns from negative-only batch |
|---|---:|---:|---|
| SSDLite320-MobileNetV3 | 0.0000 | 0.0 | No |
| RetinaNet-ResNet50-FPN-v2 | 0.1165 | 3299.83 | Yes |

RetinaNet then passed one-step training, inference, COCO evaluation, and
checkpoint serialization with both random and ImageNet-pretrained ResNet-50
backbones. Two deterministic reruns produced exactly identical loss and COCO
metrics. A checkpoint was also loaded and successfully advanced by another
optimizer step.

Decision: RetinaNet-ResNet50-FPN-v2 is the primary convolutional baseline.
SSDLite remains an integration diagnostic only.

Two independent deterministic reruns then produced identical total loss
(144.258544921875), gradient L1 (686466.7707767688), and every reported loss
component. Runtime values are excluded from this comparison.

Artifacts:

- `outputs/negative_batch_test/summary.json`
- `outputs/baseline_retinanet_smoke_pretrained/summary.json`
- `outputs/retinanet_determinism_a/summary.json`
- `outputs/retinanet_determinism_b/summary.json`
- `outputs/retinanet_resume_test/summary.json`

## Remaining work before M1 is frozen

1. Allocate CUDA-capable hardware and record its exact configuration.
2. Run one complete baseline epoch with pretrained weights.
3. Qualify the selected transformer-based comparison model.

## M1d — Transformer candidate selection

RT-DETR-R50 was selected as the candidate transformer comparison. The choice
keeps a ResNet-50 backbone, matching the convolutional baseline more closely
than the R18/R34 variants, while changing the detection formulation to
end-to-end set prediction without NMS. The original implementation reports a
640-pixel input, 42 million parameters, and 136 GFLOPs for this variant; these
published COCO figures are architecture context only, not project results.

Vanilla DETR was rejected because its official training recipe uses 300 epochs
and eight V100 GPUs, making it a poor fit for the available compute budget.
Deformable DETR was not selected because its official repository requires
compilation of custom CUDA operators. RT-DETR is available through both the
authors' official PyTorch implementation and Hugging Face Transformers, which
provides a lower-risk integration path for the existing COCO artifacts.

Status: preliminary CPU qualification passed on 2026-09-02. At diagnostic
resolution 320, RT-DETR-R50 completed a finite one-step optimization, produced
nonzero gradients, ran inference, serialized and reloaded model weights, and
completed the official COCO evaluation path. A separate batch of two
negative-only images produced classification loss 1271.9138 and gradient L1
17,097,481.73, with zero box and GIoU losses as expected. These magnitudes
reflect newly initialized four-class heads and are not performance results.

Artifacts:

- `outputs/rtdetr_smoke_320/summary.json`
- `outputs/rtdetr_negative_batch/summary.json`
- `outputs/rtdetr_coco_eval_smoke/summary.json`
- `outputs/rtdetr_determinism_a/summary.json`
- `outputs/rtdetr_determinism_b/summary.json`

Promotion remains pending until full checkpoint resume (including optimizer
state) passes, followed by a native 640-by-640 GPU epoch.

Primary references:

- <https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html>
- <https://github.com/lyuwenyu/RT-DETR>
- <https://huggingface.co/docs/transformers/en/model_doc/rt_detr>

## M2 — CPU-only scope and model selection

- Date: 2026-09-02
- Hardware: Intel Core i5-10500T, 6 physical cores / 12 threads, 31 GiB RAM.
- Decision: GPU-dependent RetinaNet/RT-DETR comparisons were archived after
  measured one-epoch estimates exceeded approximately 9.5 hours per model.
- Selected model: COCO-pretrained YOLO11n, 2.59 million parameters, 320-pixel
  input, batch size 8.

The low-resolution Faster R-CNN MobileNetV3 qualification learned from
negative-only batches but required a median 2.24 seconds per two-image batch.
After 500 optimization steps it reached 1.25% COCO mAP and 3.29% AP50 on a
fixed 500-image validation subset. This was a pilot, not a test-set result.

A first YOLO11n pilot intentionally exposed a sampling flaw in the library's
`fraction` option: because the dataset is path-sorted, the selected 2% contained
only China Drone images. That run was stopped and excluded. Explicit lists were
then generated with equal domain quotas and domain-specific negative rates.

The corrected 1,400-image pilot (200 per domain) completed five epochs in 8.8
minutes of CPU training and reached 1.75% mAP / 5.35% AP50 on all 3,661
validation images. Since the learning curve was still improving, the frozen
primary budget was increased before any internal-test inference to 2,800 images
(400 per domain), 10 epochs, or 28,000 image presentations. The internal test
remained untouched during this decision.

## M3 — Frozen CPU baseline and internal-test evaluation

- Date completed: 2026-09-03
- Status: complete
- Model: COCO-pretrained YOLO11n, 2,590,620 parameters.
- Training: 2,800 images (400 per domain), 925 negatives, 10 epochs, batch 8,
  320-pixel input, AdamW, seed 2026, deterministic mode.
- Logged cumulative training time: 1,709.61 s (28.49 min).
- Final validation: 3.007% mAP and 8.573% AP50 on 3,661 images.
- Frozen internal test: 3.261% mAP, 8.512% AP50, and 1.818% AP75 on 3,703
  images and 5,486 instances.
- Domain macro/worst mAP: 3.469% / 0.420%; worst domain: Norway.
- Class mAP: D00 3.827%, D10 3.010%, D20 5.458%, D40 0.747%.
- Scale AP: small 0.262%, medium 2.705%, large 3.625%.
- Negative images: 1,414; false-positive rate at score 0.25: 0.919%.
- Model-stage latency: 23.02 ms/image (43.44 images/s).
- Complete evaluator: 162.10 s, peak RSS 1,392,780 KiB (1.33 GiB).

The first two attempts to evaluate by passing the complete pathname list to
Ultralytics were stopped because its loader exhausted host memory. They wrote
no scientific result. The final evaluator invokes inference on one pathname at
a time, uses one thread and batch size one, persists predictions before COCO
slicing, and remained below 1.33 GiB. A separate 20-image smoke directory was
used only to verify this memory correction and is excluded from the paper.

Frozen artifacts:

- `outputs/yolo11n_joint_cpu_final_seed2026/weights/last.pt`
- `outputs/yolo11n_joint_cpu_final_seed2026/results.csv`
- `outputs/yolo11n_joint_cpu_final_seed2026/test_evaluation/evaluation.json`
- `outputs/yolo11n_joint_cpu_final_seed2026/test_evaluation/predictions.json`
- `outputs/cpu_protocol_2800/protocol.json`
