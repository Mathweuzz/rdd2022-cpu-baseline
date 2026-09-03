# Results provenance

The manuscript reports one frozen CPU-only run. The internal test was evaluated
only after the model, subset, optimizer, epoch count, and final-checkpoint rule
were fixed. Qualification pilots are documented separately in
`experiments/EXPERIMENT_LOG.md` and are not test-set baselines.

## Frozen configuration

- Seed: 2026; deterministic mode enabled.
- Model: COCO-pretrained YOLO11n; 2,590,620 parameters.
- Data: 2,800 training images, exactly 400 per domain; 925 negative images.
- Optimization: 10 epochs, batch 8, 320-by-320 input, AdamW, learning rate
  0.00125, final factor 0.05, weight decay 0.0005, one warm-up epoch.
- Selection: final epoch (`last.pt`), with no early stopping.
- Test: all 3,703 internal-test images; COCO maxDets 100; score serialization
  floor 0.001; no domain-specific thresholding.
- Runtime: Intel Core i5-10500T CPU, no CUDA; Python 3.14.7, PyTorch 2.10.0,
  TorchVision 0.25.0, Ultralytics 8.3.203.

## Low-memory evaluation command

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=.deps \
YOLO_CONFIG_DIR="$PWD/outputs/ultralytics-config" \
MPLCONFIGDIR="$PWD/outputs/matplotlib-config" \
python experiments/evaluate_yolo_checkpoint.py \
  --model outputs/yolo11n_joint_cpu_final_seed2026/weights/last.pt \
  --split test \
  --output outputs/yolo11n_joint_cpu_final_seed2026/test_evaluation \
  --imgsz 320 --batch 1 --workers 0 --threads 1
```

## SHA-256

```text
1029297eacb1305144eef1fc5d6c7ef69e43ac40dfde980198d826ef1a2c93f5  outputs/cpu_protocol_2800/protocol.json
76d0e62c0c5a4f17d69800ff09ae9f4ee3d330e93a229096716d38267b7142b3  outputs/yolo11n_joint_cpu_final_seed2026/weights/last.pt
f907a4ddd7ec947c964f465783002d1618b6b50115b4a9eba737b2729df87587  outputs/yolo11n_joint_cpu_final_seed2026/test_evaluation/evaluation.json
70019b1805ea3c8f1f7e1ca33421a07635e0f7fa881b6758af68e04a4fc5206a  outputs/yolo11n_joint_cpu_final_seed2026/test_evaluation/predictions.json
```

Tracked publication artifacts:

```text
7c40dc5832c0824e0799f23eb177e6dd207fa2cac688a7851118a81415028136  artifacts/frozen_protocol.json
f907a4ddd7ec947c964f465783002d1618b6b50115b4a9eba737b2729df87587  artifacts/test_evaluation.json
0e3318364de33dd2d6ab15396ec4d3de9e3ff09d77e60a66e91e2be1f018923a  artifacts/training_history.csv
3c16438209121acc912de91f5251b70147d7369c4d866f39c3aec3168e1ca380  artifacts/training_config.yaml
d71a7707325c3e20825816bfe270b978bcb9298e94394b2e3198ac4c204308ae  paper/main.tex
1792dd095fd7fc4214cc5774c4f08668633217e6d76328ee3aba61980ba715a3  paper/main.pdf
```

The tracked copy `artifacts/test_evaluation.json` is the machine-readable source
for every metric in the Results, Domain-Wise Analysis, and Conclusion sections.
`artifacts/training_history.csv` is the source for training time, losses, and
final validation metrics. Large predictions, datasets, dependency caches, and
weights are deliberately excluded from Git history.
