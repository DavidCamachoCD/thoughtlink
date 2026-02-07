# ThoughtLink - Docker image for Ubuntu AMD64 + NVIDIA RTX 4060
# CUDA 12.4 + Python 3.12 + PyTorch with GPU support
#
# Build:   docker build -t thoughtlink .
# Run:     docker run --gpus all -it thoughtlink
# Jupyter: docker compose up jupyter

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    UV_LINK_MODE=copy

# System dependencies (single layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    curl \
    # MuJoCo / OpenGL headless rendering
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglfw3 \
    libglfw3-dev \
    libosmesa6-dev \
    patchelf \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install UV (pinned version for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:0.6 /uv /uvx /bin/

WORKDIR /app

# Dependency install layer (cached unless pyproject.toml/uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --python 3.12

# Project source
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY tests/ tests/
COPY notebooks/ notebooks/

# Healthcheck: verify Python + thoughtlink importable
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD uv run python -c "import thoughtlink" || exit 1

CMD ["uv", "run", "python", "-c", "import thoughtlink; print(f'ThoughtLink v{thoughtlink.__version__} ready')"]

# --- Jupyter stage ---
FROM base AS jupyter
EXPOSE 8888
CMD ["uv", "run", "jupyter", "notebook", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root", "--NotebookApp.token=''"]

# --- Streamlit stage ---
FROM base AS streamlit
EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", \
     "src/thoughtlink/viz/dashboard.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
