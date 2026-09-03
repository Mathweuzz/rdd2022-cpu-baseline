PYTHON ?= python

.PHONY: help install install-dev download eda prepare validate protocol train test paper check

help:
	@echo "install      Install data/EDA dependencies"
	@echo "install-dev  Install experiment and development dependencies"
	@echo "download     Download and unpack the official RDD2022 archive"
	@echo "eda          Generate audited EDA tables and figures"
	@echo "prepare      Build sequence-blocked COCO/YOLO data"
	@echo "validate     Validate every generated image, label, and split"
	@echo "protocol     Build the frozen 2,800-image CPU protocol"
	@echo "train        Reproduce the frozen YOLO11n CPU run"
	@echo "test         Run the lightweight test suite"
	@echo "paper        Compile the six-page IEEE manuscript"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-experiments.txt pytest==9.1.1 ruff==0.12.12

download:
	$(PYTHON) download_rdd2022.py --extract

eda:
	$(PYTHON) eda_rdd2022.py

prepare:
	$(PYTHON) prepare_rdd2022.py --extract-images

validate:
	$(PYTHON) validate_rdd2022.py

protocol:
	$(PYTHON) experiments/build_cpu_yolo_protocol.py \
		--train-images 2800 --seed 2026 --output outputs/cpu_protocol_2800

train:
	$(PYTHON) experiments/train_yolo_cpu.py

test:
	$(PYTHON) -m pytest -q

paper:
	$(MAKE) -C paper

check: test paper
