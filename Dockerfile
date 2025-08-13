FROM rapidsai/notebooks:24.12a-cuda12.0-py3.11

ENV DEBIAN_FRONTEND=noninteractive

# Add jupyterlab explicitly and keep everything in base
RUN mamba install -y -n base -c conda-forge --freeze-installed \
    pandas jupyterlab ipykernel \
    scipy scikit-learn pyarrow fsspec python-dotenv mlflow cuopt \
 && mamba clean -y --all

RUN python -m pip install --no-cache-dir \
    azure-ai-ml ortools adlfs pre-commit nbstripout

RUN python -m ipykernel install --user --name=rpd-sk --display-name "rpd-sk"

COPY . /home/rapids/notebooks/ml-for-gpu
WORKDIR /home/rapids/notebooks/ml-for-gpu