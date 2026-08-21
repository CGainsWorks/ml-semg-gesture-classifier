# ml-semg-gesture-classifier
#
# `make demo` goes from a clean checkout to every number in the README.
# `make smoke` does the same in two minutes on synthetic data, no download.
# `docker compose up` does `make demo` in a container.

PYTHON ?= python
CONFIG ?= config/default.yaml
CI_CONFIG ?= config/ci.yaml

.DEFAULT_GOAL := help
.PHONY: help setup data dataset-report baseline deep train experiments deploy \
        export bench estimate control-loop calibrate evaluate figures test \
        lint smoke demo docker-build docker-up docker-down clean clean-all

help:
	@echo "make setup           install the pinned dependencies"
	@echo "make data            download Ninapro DB5 (about 200 MB) and checksum it"
	@echo "make smoke           whole pipeline on synthetic data, no download, ~2 min"
	@echo "make baseline        train the classical models, all protocols"
	@echo "make deep            train the TensorFlow and PyTorch models, all protocols"
	@echo "make train           baseline + deep"
	@echo "make deploy          train the models that get shipped, on every subject"
	@echo "make export          quantise to TFLite int8 and ONNX int8, and score both"
	@echo "make bench           measure single-window latency on this machine"
	@echo "make estimate        on-target latency estimates (not measurements)"
	@echo "make control-loop    stream the recordings through the deployed model"
	@echo "make calibrate       fit the model to one wearer from their demonstrations"
	@echo "make evaluate        figures, tables and analysis/db5/EVALUATION.md"
	@echo "make experiments     the architecture and latency ablations"
	@echo "make test            run the test suite"
	@echo "make lint            ruff"
	@echo "make demo            everything above, end to end, from nothing"
	@echo "make docker-up       the same thing in Docker"
	@echo "make clean           remove trained models, caches and exports"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

data:
	$(PYTHON) -m src.data.download --verify

dataset-report:
	$(PYTHON) -m src.data.build_report --config $(CONFIG)

baseline:
	$(PYTHON) -m src.train --config $(CONFIG) --model classical_lda --protocol all
	$(PYTHON) -m src.train --config $(CONFIG) --model classical_rf --protocol all
	# The RBF SVM is within-subject only; scripts/run_experiments.sh says why.
	$(PYTHON) -m src.train --config $(CONFIG) --model classical_svm --protocol within_subject

deep:
	$(PYTHON) -m src.train --config $(CONFIG) --model tensorflow --protocol all
	$(PYTHON) -m src.train --config $(CONFIG) --model pytorch --protocol all

train: baseline deep

deploy:
	$(PYTHON) -m src.train --config $(CONFIG) --model tensorflow --protocol deploy
	$(PYTHON) -m src.train --config $(CONFIG) --model pytorch --protocol deploy

export: deploy
	$(PYTHON) -m src.quantise.export --config $(CONFIG)

bench:
	$(PYTHON) -m src.embedded.benchmark --config $(CONFIG) --label host

estimate:
	$(PYTHON) -m src.embedded.estimate_report --config $(CONFIG)

control-loop:
	$(PYTHON) -m src.embedded.control_loop --config $(CONFIG)

# Learning by demonstration: the fitting appointment, as a command.
# SUBJECT=3 make calibrate. --holdout trains the "before" model here, excluding
# the wearer, which is the only version whose before-and-after means anything.
SUBJECT ?= 1
calibrate:
	$(PYTHON) -m src.calibrate --config $(CONFIG) --subject $(SUBJECT) --holdout

experiments:
	$(PYTHON) -m experiments.architecture_comparison --config $(CONFIG)
	$(PYTHON) -m experiments.latency_accuracy_tradeoff --config $(CONFIG)

evaluate:
	$(PYTHON) -m src.evaluate --config $(CONFIG)

figures:
	$(PYTHON) -m src.eval.make_figures --config $(CONFIG)

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests experiments

# Everything, on the real data. Hours, and it needs the download.
# `bash scripts/run_demo.sh` is the same thing without the download step, and
# falls back to the synthetic run if the data is absent.
demo: data
	bash scripts/run_experiments.sh

# Everything, on synthetic data, in about two minutes. No download, no GPU.
# The accuracy it prints is a wiring check and is labelled as one.
smoke:
	bash scripts/run_smoke.sh

docker-build:
	docker compose build

docker-up:
	docker compose up

docker-down:
	docker compose down -v

clean:
	rm -rf artifacts
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

# Also removes the downloaded data. `make data` fetches it again.
clean-all: clean
	rm -rf data/raw .pytest_cache .ruff_cache
