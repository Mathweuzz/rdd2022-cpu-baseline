import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_metrics_and_protocol_are_consistent():
    metrics = json.loads((ROOT / "artifacts/test_evaluation.json").read_text())
    protocol = json.loads((ROOT / "artifacts/frozen_protocol.json").read_text())
    assert metrics["status"] == "complete"
    assert metrics["images"] == protocol["test_images"] == 3703
    assert metrics["instances"] == sum(protocol["test_instances"].values()) == 5486
    assert metrics["parameters"] == protocol["parameters"] == 2_590_620
    assert set(metrics["per_domain"]) == set(protocol["test_images_by_domain"])
    assert abs(metrics["metrics"]["map_50_95"] - 0.03260629785463379) < 1e-15


def test_training_history_has_frozen_final_epoch():
    with (ROOT / "artifacts/training_history.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert int(rows[-1]["epoch"]) == 10
    assert abs(float(rows[-1]["time"]) - 1709.61) < 1e-9
    assert abs(float(rows[-1]["metrics/mAP50-95(B)"]) - 0.03007) < 1e-9


def test_paper_build_inputs_are_tracked_and_portable():
    makefile = (ROOT / "paper/Makefile").read_text()
    source = (ROOT / "paper/main.tex").read_text()
    assert "figures/overview.png" in makefile
    assert "../outputs/" not in makefile
    assert "\\graphicspath{{figures/}}" in source
    assert (ROOT / "paper/figures/overview.png").is_file()


def test_readme_pipeline_order_matches_dependencies():
    readme = (ROOT / "README.md").read_text()
    commands = ["make download", "make eda", "make prepare", "make validate", "make protocol"]
    positions = [readme.index(command) for command in commands]
    assert positions == sorted(positions)
