**Devcontainer / Docker usage**

- Build CPU image:
```
docker build --build-arg BASE_IMAGE=python:3.11-slim -t cellstates:dev .
```

- Build CUDA image (Linux hosts with NVIDIA toolkit):
```
docker build --build-arg BASE_IMAGE=nvidia/cuda:12.2.0-devel-ubuntu22.04 -t cellstates:cuda .
```

- Run CPU image:
```
docker run --rm cellstates:dev
```

- Run CUDA image (on Linux with NVIDIA Container Toolkit):
```
docker run --rm --gpus all cellstates:cuda python -c "import torch; print(torch.cuda.is_available())"
```

- Open in VS Code: install the Remote - Containers (Dev Containers) extension, then choose `Dev Containers: Reopen in Container`.

Notes:
- On macOS, Apple MPS is not available inside Docker. Use a native Conda/venv environment for MPS.
- For TPUs, use cloud TPU VMs (GCP) and VS Code Remote-SSH.