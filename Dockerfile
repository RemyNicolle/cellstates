ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    python3-dev \
    pkg-config \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN python -m pip install --upgrade pip setuptools wheel cython numpy

COPY . /workspace

RUN pip install -e .

CMD ["python", "-c", "import cellstates; print('cellstates OK')"]
