# RAPIDS notebooks with pandas 2.x support
FROM rapidsai/notebooks:24.12a-cuda12.0-py3.12

ENV DEBIAN_FRONTEND=noninteractive

# Add extras, but keep RAPIDS pins intact
RUN mamba install -y -n base -c conda-forge --freeze-installed \
      jupyterlab ipykernel scikit-learn pyarrow fsspec python-dotenv mlflow \
 && mamba clean -y --all

# cuOpt server + client (Py3.12-supported builds)
RUN mamba install -y -n base --override-channels -c rapidsai -c nvidia -c conda-forge \
      --freeze-installed --strict-channel-priority \
      cuopt-server=25.* cuopt-sh-client=25.* \
 && mamba clean -y --all \
 && python -m ipykernel install --sys-prefix --name=rpd-sk --display-name "rpd-sk"