# Reproduces the pipeline without touching the host Python.
#
# CPU only, deliberately. A CUDA base image is several gigabytes and would make
# the container unbuildable on a machine without an NVIDIA runtime, which
# defeats the point of shipping one. This model trains on a CPU in minutes for
# the synthetic smoke run and in a few hours for the full DB5 sweep; pass
# --device on the training commands if you want the host's GPU and have the
# NVIDIA container toolkit installed.

FROM python:3.12-slim-bookworm

# libgomp1 for the OpenMP runtime that both torch and scikit-learn link
# against; curl so the dataset download works inside the container.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MPLBACKEND=Agg

WORKDIR /app

# Dependencies before source, so an edit to src/ does not reinstall torch.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 \
    && pip install -r requirements-dev.txt

COPY src/ ./src/
COPY tests/ ./tests/
COPY experiments/ ./experiments/
COPY config/ ./config/
COPY scripts/ ./scripts/
# The dataset card and the archive digests, so a `make demo` inside the
# container can verify what it downloaded against the committed checksums
# rather than trusting whatever it got.
COPY data/DATASET_CARD.md data/DB5_CHECKSUMS.txt ./data/
COPY Makefile pyproject.toml ./

# Run as a normal user. The mounted artifacts and data directories are created
# and chowned here so a bind mount from the host is writable.
RUN mkdir -p artifacts analysis benchmarks data/raw \
    && useradd --create-home --uid 1000 semg \
    && chown -R semg:semg /app
USER semg

# The synthetic smoke run: no download, no GPU, about two minutes. Override to
# run the real thing:
#   docker run --rm -v "$PWD/data:/app/data" ... make demo
CMD ["make", "smoke"]
