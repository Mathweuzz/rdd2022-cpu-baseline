"""Safe loading helpers for Ultralytics models released without pickle."""

from __future__ import annotations

from pathlib import Path

from safetensors.torch import load_file
from ultralytics import YOLO


CLASS_NAMES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}


def load_yolo(model_path: Path | str, config_path: Path | str | None = None) -> YOLO:
    """Load a native checkpoint or a safe tensor file plus architecture YAML."""
    path = Path(model_path)
    if path.suffix != ".safetensors":
        return YOLO(str(path))
    config = Path(config_path) if config_path else path.with_suffix(".yaml")
    if not config.is_file():
        raise FileNotFoundError(
            f"Architecture YAML not found: {config}. Download it beside {path.name}."
        )
    model = YOLO(str(config), task="detect")
    model.model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    model.model.names = CLASS_NAMES.copy()
    return model
