# GPU Acceleration Demo

Demonstrate 20–50× speedups by migrating CPU workflows (pandas, scikit-learn, OR-Tools) to GPU (cuDF, cuML, cuOpt) on Azure ML with minimal code changes.

## 🎯 Objectives

- **ETL + ML**: pandas → cuDF, sklearn → cuML on Avazu CTR dataset
- **Optimization**: OR-Tools → cuOpt on VRPTW (Gehring & Homberger RC2)
- **Speedup**: 20-50× acceleration across read, ETL, fit, predict, solve stages
- **Model Parity**: AUC/logloss within ±0.5%, feasible VRPTW solutions
- **Migration Effort**: ≤5 lines changed per notebook

## 🏗️ Architecture

- **Compute**: Azure ML Compute Instance (NC_A100)
- **Environment**: Single RAPIDS-based environment for CPU/GPU workflows
- **Data**: Azure ML Data Assets (uri_file, ADLS Gen2)
- **Auth**: DefaultAzureCredential + .env configuration

## 📊 Datasets

### Avazu CTR (ETL + ML)
- **Format**: Single Parquet file (~40M rows)
- **Target**: Binary click prediction (0/1)
- **Features**: hour_of_day, frequency encoding for categoricals
- **Models**: Logistic Regression (sklearn → cuML)

### Gehring & Homberger VRPTW (Optimization)
- **Format**: Customers Parquet + Parameters JSON
- **Schema**: customer_id, x, y, demand, tw_start, tw_end, service_time
- **Solvers**: OR-Tools → cuOpt
- **Instances**: RC2 (200-1000 customers)

## 🚀 Quick Start

### 1. Environment Setup


Update `.env` with your Azure ML details

#### Download datasets:

```bash
source .env
mkdir data
mkdir data/avazu
mkdir data/vrptw/homberger/c2
mkdir data/vrptw/homberger/rc2

# Avazu CTR dataset - you will need to accept the competition rules and create a kaggle accoutn to access to full ~40M row dataset
curl -L -C - --fail \
--user "$KAGGLE_USERNAME:$KAGGLE_KEY" \
"https://www.kaggle.com/api/v1/competitions/data/download/avazu-ctr-prediction/train.gz" \
-o data/avazu/avazu-ctr.gz

# Avazu CTR dataset - 50k row subsample for testing
curl -L -o data/avazu/avazu-ctr-50k.zip \
https://www.kaggle.com/api/v1/datasets/download/gauravduttakiit/avazu-ctr-prediction-with-random-50k-rows


# Homberger 100 customer instance (RC2)
wget -c -O data/vrptw/homberger/rc2/homberger_1000_customer_instances.zip \ 
"https://www.sintef.no/globalassets/project/top/vrptw/homberger/1000/homberger_1000_customer_instances.zip"

# Homberger 200 customer instance - smaller for testing (C2)
wget -c -O data/vrptw/homberger/c2/homberger_200_customer_instances.zip \
"https://www.sintef.no/globalassets/project/top/vrptw/homberger/200/homberger_200_customer_instances.zip"  
```

#### Upload datasets to AzureML
Upload to AzureML and register as a Data Asset
```bash
./utils/upload_to_azureml.sh <LOCAL_PATH> "<DESCRIPTION>"   
```

#### Register the Azure ML environment:

```bash
# Configure Azure CLI (if not already done)
az login
az account set --subscription "your-subscription-id"

# Register environment
```bash 
./utils/register_env.sh  
```

### 3. Run Notebooks

In Azure ML Studio:

1. Open Compute Instance with NC_A100
2. Select kernel: **AzureML: rapids-sklearn-cu12**
3. Run notebooks:
   - `notebooks/01_etl_ml_cpu_vs_gpu.ipynb`
   - `notebooks/02_optimization_vrptw.ipynb`

## 📁 Directory Structure

```
ml-for-gpu/
├── env/
│   ├── conda-rapids-sklearn.yml    # Conda dependencies
│   └── environment.yml             # Azure ML environment spec
├── notebooks/
│   ├── 01_etl_ml_cpu_vs_gpu.ipynb  # ETL + ML comparison
│   └── 02_optimization_vrptw.ipynb # VRPTW optimization comparison
├── utils/
│   ├── register_env.sh             # Environment registration script
│   ├── data_access.py              # Azure ML data utilities
│   ├── timing.py                   # Performance timing utilities
│   ├── diff_cells.py               # Code diff analysis
│   └── homberger_to_parquet.py     # VRPTW data converter
├── data/                           # Local data staging
│   └── README.md
├── .env                           # You need to make/populate this
└── README.md
```

## 🛠️ Utilities

### Data Access (`utils/data_access.py`)
- Resolve Azure ML data assets to ABFSS paths
- Create storage options with DefaultAzureCredential
- Handle authentication and path resolution

### Timing (`utils/timing.py`)
- GPU-aware performance measurement with CUDA synchronization
- CPU thread limiting for fair comparisons
- Consistent timing across CPU/GPU workflows

### Code Diff Analysis (`utils/diff_cells.py`)
- Count lines changed between CPU and GPU implementations
- Demonstrate minimal migration effort
- Generate diff reports for documentation

### Data Conversion (`utils/homberger_to_parquet.py`)
- Convert Homberger VRPTW instances (.txt) to Parquet + JSON
- Extract from ZIP archives
- Generate Azure ML-compatible data assets

## 🔧 Environment Details

**Base Image**: [`rapidsai/rapidsai:cuda11.8-runtime-ubuntu22.04-py3.10`](https://hub.docker.com/layers/rapidsai/rapidsai/cuda11.8-runtime-ubuntu22.04-py3.10/images/sha256-60e3ae97db947a237e5de571a92a37437174f983dd1c31e3cfce2b0afb45d085)

**Auto Cleanup**:

This repo uses pre-commit to automatically clear notebook outputs before commit. The [pre-commit-config.yaml](./pre-commit-config.yaml) file has been provided for you

```bash
pip install pre-commit  
pre-commit install  
pre-commit autoupdate  
```


