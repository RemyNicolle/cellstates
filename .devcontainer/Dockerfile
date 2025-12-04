ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} as base

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    python3-dev \
    pkg-config \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Upgrade pip and install Python build/runtime tools
RUN python -m pip install --upgrade pip setuptools wheel cython numpy

# Copy project sources
COPY .. /workspace

# Install package in editable mode (builds Cython extensions)
RUN pip install -e .

CMD ["python", "-c", "import cellstates; print('cellstates OK')"]
# Build args:
#  - BASE_IMAGE: optional. If you want CUDA, set it to an NVIDIA CUDA image.
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} as base

ENV DEBIAN_FRONTEND=noninteractive

# Install typical build deps (adjust to your needs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    python3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Ensure pip/setuptools/wheel, then install Python deps necessary to build Cython extensions
RUN python -m pip install --upgrade pip setuptools wheel cython numpy

# Copy sources and install in editable mode (this will run setup.py and build extensions)
COPY . /workspace
RUN pip install -e .

# Default command for quick smoke-test
CMD ["python", "-c", "import cellstates; print('cellstates OK')"]