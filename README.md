# GPU Acceleration Demo

Demonstrate incredible performance gains by migrating common ML and OR solutions from CPU to GPU with minimal code changes.

## 🎯 Objectives

Demonstrated performance gains from GPU based libraries - with minimal migration effort.

- **ML - Classification**: pandas → cuDF, sklearn → cuML on Avazu CTR dataset
- **Optimization**: OR-Tools → cuOpt on VRPTW (Gehring & Homberger dataset)
- **Solution Parity**: AUC/logloss within ±0.5%, feasible VRPTW solutions

## 📊 Datasets

### Avazu CTR (ETL + ML)
- **Format**: Single Parquet file (~40M rows) or 50k sample
- **Target**: Binary click prediction (0/1)
- **Features**: hour_of_day, frequency encoding for categoricals
- **Models**: Logistic Regression (sklearn → cuML)

### Gehring & Homberger VRPTW (Optimization)
- **Format**: ZIP archives with .txt instance files
- **Schema**: customer_id, x, y, demand, tw_start, tw_end, service_time
- **Solvers**: OR-Tools (CPU) → cuOpt (GPU)
- **Instances**: C2 (200 customers), RC2 (1000 customers)

## Results Summary

| Dataset | GPU Type | Optimization Speedup | ML Classification Speedup|
:--------:|:--------:|:--------:|:--------:|
Sample  | Nvidia T4 | 2.9x | 1.4x
Sample | Nvidia A100 | 9.1x | 2.0x 
**Full** | **Nvidia A100** |**1.9x++** | **8.5x**

_*Sample = Avazu CTR 50k & Gehring & Homberger VRPTW 200 customer_  
_**Full = Avazu CTR Full & Homberger VRPTW 100 customer_
_++ = Superior result quality (AUC, Solve Quality, etc.)_

## 🚀 Quick Start

#### Environment Setup

Update `.env` according to the provided sample

#### Download datasets:

```bash
source .env
mkdir classification/data
mkdir optimization/data

# Avazu CTR dataset - you will need to accept the competition rules and create a kaggle accoutn to access to full ~40M row dataset
curl -L -C - --fail \
--user "$KAGGLE_USERNAME:$KAGGLE_KEY" \
"https://www.kaggle.com/api/v1/competitions/data/download/avazu-ctr-prediction/train.gz" \
-o classification/data/avazu-ctr.gz

# Avazu CTR dataset - 50k row subsample for testing
curl -L -o classification/data/avazu-ctr-50k.zip \
https://www.kaggle.com/api/v1/datasets/download/gauravduttakiit/avazu-ctr-prediction-with-random-50k-rows


# Homberger 1000 customer instance (RC2)
wget -c -O optimization/data/homberger_1000_customer_instances.zip "https://www.sintef.no/globalassets/project/top/vrptw/homberger/1000/homberger_1000_customer_instances.zip"

# Homberger 200 customer instance - smaller for testing (C2)
wget -c -O optimization/data/homberger_200_customer_instances.zip \
"https://www.sintef.no/globalassets/project/top/vrptw/homberger/200/homberger_200_customer_instances.zip"  
```

#### Setup Compute Instance on AzureML (GPU accelerated)

1. Create NC_A100 (or similar GPU) Compute Instance
2. Launch VS Code from Azure ML Studio
3. Clone this repository in the terminal
4. Navigate to the project directory

#### Setup Notebook Kernels

```bash
cd <path to repo>
conda install mamba -n base -c conda-forge

# Create environment and install kernels
mamba env create -f classification/rapids-sk.yml
mamba run -n rapids-sk python -m ipykernel install --user --name rapids-sk --display-name "rapids-sk"

mamba env create -f optimization/cuopt-or.yml
mamba run -n cuopt-or python -m ipykernel install --user --name cuopt-or --display-name "cuopt-or"

# Validate exists
jupyter kernelspec list
```
__Note: You may have to reload window for new kernels to showup in your notebook__

### 5. Run Notebooks

1. **Classification**: Open `classification/ml_cpu_vs_gpu.ipynb` and select `rapids-sk` kernel
2. **Optimization**: Open `optimization/02_optimization_vrptw.ipynb` and select `cuopt-or` kernel

---

## 📁 Directory Structure

```
ml-for-gpu/
├── .env                           # Azure ML credentials (create from .sample.env)
├── .sample.env                    # Template for environment variables
├── classification/
│   ├── ml_cpu_vs_gpu.ipynb        # ETL + ML CPU vs GPU comparison
│   ├── rapids-sk.yml              # Conda environment: RAPIDS + scikit-learn
│   ├── data/                      # Avazu CTR dataset storage
│   └── utils/
│       └── timing.py              # Performance measurement utilities
├── optimization/
│   ├── optimization_vrptw.ipynb # VRPTW optimization CPU vs GPU
│   ├── cuopt-or.yml               # Conda environment: cuOpt + OR-Tools  
│   ├── data/                      # Homberger VRPTW dataset storage
│   └── utils/
│       ├── timing.py              # Performance measurement utilities
│       └── homberger_to_parquet.py # VRPTW data parsing utilities
└── README.md
```

## 🛠️ Key Components

### Conda Environments

**RAPIDS + Scikit-learn** (`classification/rapids-sk.yml`):
- RAPIDS 24.10 (cuDF, cuML) with CUDA 12.0 support
- scikit-learn, pandas, numpy for CPU baselines
- Jupyter kernel for classification notebooks

**cuOpt + OR-Tools** (`optimization/cuopt-or.yml`):
- cuOpt 25.08.00 for GPU optimization
- OR-Tools for CPU optimization baselines  
- RAPIDS 24.10 for DataFrame operations
- Jupyter kernel for optimization notebooks

### Utilities

**Performance Timing** (`utils/timing.py`):
- GPU-aware measurement with CUDA synchronization
- CPU thread limiting for fair comparisons
- Consistent timing across CPU/GPU workflows

**VRPTW Data Processing** (`optimization/utils/homberger_to_parquet.py`):
- Parse Homberger .txt instance files
- Extract from ZIP archives
- Convert to pandas/cuDF compatible formats

## 🤝 Contributing

1. Use pre-commit hooks to clear notebook outputs:
```bash
pip install pre-commit
pre-commit install
```