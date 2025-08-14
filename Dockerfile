FROM rapidsai/notebooks:24.10a-cuda12.2-py3.10

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Keep everything in base; don't break RAPIDS deps
RUN mamba install -y -n base -c conda-forge --freeze-installed \
      "pandas>=2" ipykernel scipy scikit-learn pyarrow fsspec python-dotenv mlflow \
 && mamba clean -y --all

# cuOpt server + client (you only need client if you call a sidecar server;
# if you're using the in-process solver APIs, this still doesn't hurt)
RUN python -m pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com \
      cuopt-server-cu12 cuopt-sh-client

# Register a kernel inside the container (nice-to-have, not required for docker-kernel mode)
RUN python -m ipykernel install --sys-prefix --name=cu-sk --display-name="cu-sk"

COPY . /home/rapids/notebooks/ml-for-gpu
WORKDIR /home/rapids/notebooks/ml-for-gpu