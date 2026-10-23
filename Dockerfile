# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm AS cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src/traffic_violation/__init__.py ./src/traffic_violation/__init__.py
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install -e ".[dev,web]"

COPY src ./src
COPY configs ./configs
COPY data ./data
COPY demo ./demo
COPY scripts ./scripts
COPY tests ./tests

CMD ["sh", "-lc", "pytest -v -m 'not gpu' && python demo/00_data.py && python demo/01_config.py"]


FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src/traffic_violation/__init__.py ./src/traffic_violation/__init__.py
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install -e ".[dev,ml,vlm,quant]"

COPY src ./src
COPY configs ./configs
COPY data ./data
COPY demo ./demo
COPY scripts ./scripts
COPY tests ./tests

CMD ["sh", "-lc", "pytest -v -m 'not gpu' && python demo/04_detect.py && python demo/05_tracking.py"]

