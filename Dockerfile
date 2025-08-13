FROM rapidsai/rapidsai:cuda11.8-runtime-ubuntu22.04-py3.10

ENV DEBIAN_FRONTEND=noninteractive

# Add jupyterlab explicitly and keep everything in base
RUN mamba install -y -n base -c conda-forge --freeze-installed \
    jupyterlab ipykernel \
    scipy pandas scikit-learn pyarrow fsspec python-dotenv mlflow \
 && mamba clean -y --all

RUN python -m pip install --no-cache-dir \
    azure-ai-ml ortools adlfs pre-commit nbstripout

RUN python -m ipykernel install --user --name=rapids-k --display-name "rapids-k"


WORKDIR /workspace
EXPOSE 8888
CMD ["bash","-lc","jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"]
