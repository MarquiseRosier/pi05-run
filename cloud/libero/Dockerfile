FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MUJOCO_GL=egl

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libglvnd0 \
    libglx0 \
    libopengl0 \
    libosmesa6-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    pkg-config \
    python3.12 \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/opt/lerobot/bin:${PATH}" \
    VIRTUAL_ENV=/opt/lerobot

RUN uv venv /opt/lerobot --python 3.12 && \
    uv pip install --python /opt/lerobot/bin/python --torch-backend cu128 \
      "lerobot[evaluation,libero,pi]" \
      hf-transfer

WORKDIR /workspace
COPY run_pi05_libero.sh /workspace/run_pi05_libero.sh
RUN chmod +x /workspace/run_pi05_libero.sh

CMD ["/bin/bash"]
