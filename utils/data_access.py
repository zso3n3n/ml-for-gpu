import os
import zipfile
import gzip
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
import pandas as pd
import fsspec

load_dotenv()

def resolve_abfss(asset_name: str, version: str|int = None):
    """Resolve Azure ML data asset to ABFSS path and storage account."""
    mlc = MLClient(
        DefaultAzureCredential(),
        os.getenv("SUBSCRIPTION_ID"),
        os.getenv("RESOURCE_GROUP"),
        os.getenv("WORKSPACE_NAME"),
    )
    ds_name = os.getenv("AZUREML_DATASTORE")
    
    if version is None:
        # Get latest version
        asset_list = mlc.data.list(name=asset_name)
        asset = max(asset_list, key=lambda a: int(a.version))
    else:
        asset = mlc.data.get(name=asset_name, version=str(version))
    
    ds = mlc.datastores.get(ds_name)
    account, container = ds.account_name, ds.container_name
    prefix = asset.path.split("/paths/", 1)[1].strip("/")
    return f"abfss://{container}@{account}.dfs.core.windows.net/{prefix}", account

def storage_options(account_name: str):
    """Create storage options for Azure blob access."""
    return {"account_name": account_name, "credential": DefaultAzureCredential()}

def download_and_extract_avazu(output_csv_path: str = "data/avazu_train.csv"):
    """
    Download and extract Avazu data from Azure ML data asset to local CSV.
    Handles both .zip and .gz files dynamically.
    
    Args:
        output_csv_path: Local path where to save the extracted CSV
        
    Returns:
        str: Path to the extracted CSV file
    """
    # Get asset path and storage options
    asset_name = os.getenv("AZUREML_DATA_ASSET_CTR", "avazu-ctr")
    zip_path, account = resolve_abfss(asset_name)
    storage_opts = storage_options(account)
    
    # Create output directory
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading Avazu data from Azure ML asset: {asset_name}")
    print(f"📁 Will extract to: {output_path}")
    
    # Create temporary directory for download
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the file
        print("📥 Downloading compressed file...")
        fs = fsspec.filesystem("abfss", **storage_opts)
        
        # Determine file type from the remote path
        if zip_path.endswith('.gz'):
            temp_file = Path(temp_dir) / "avazu.gz"
            print("📦 Detected gzip format")
        elif zip_path.endswith('.zip'):
            temp_file = Path(temp_dir) / "avazu.zip"
            print("📦 Detected zip format")
        else:
            # Default to zip if unclear
            temp_file = Path(temp_dir) / "avazu.zip"
            print("📦 Assuming zip format")
        
        fs.download(zip_path, str(temp_file))
        print(f"✅ Downloaded {temp_file.stat().st_size / 1024**2:.1f} MB")
        
        # Extract based on file type
        if str(temp_file).endswith('.gz'):
            print("📦 Extracting from gzip...")
            with gzip.open(temp_file, 'rt') as gz_file:
                with open(output_path, 'w') as out_file:
                    out_file.write(gz_file.read())
        else:
            print("📦 Extracting from zip...")
            with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                # Find the CSV file
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                if not csv_files:
                    raise ValueError("No CSV files found in zip archive")
                
                # Extract the first CSV file
                csv_file = csv_files[0]
                zip_ref.extract(csv_file, temp_dir)
                extracted_csv = Path(temp_dir) / csv_file
                
                # Copy to final location
                import shutil
                shutil.copy(extracted_csv, output_path)
    
    print(f"✅ Extraction complete: {output_path}")
    return str(output_path)

def load_avazu_csv(csv_path: str, use_cudf: bool = False, sample_frac: float = None):
    """
    Load Avazu CSV with optimized dtypes.
    
    Args:
        csv_path: Path to the CSV file
        use_cudf: Whether to use cuDF instead of pandas
        sample_frac: Fraction of data to sample (for testing)
        
    Returns:
        DataFrame with Avazu data
    """
    # Import appropriate library
    if use_cudf:
        try:
            import cudf as df_lib
            print("📊 Loading with cuDF")
        except ImportError:
            print("⚠️ cuDF not available, falling back to pandas")
            import pandas as df_lib
    else:
        import pandas as df_lib
        print("📊 Loading with pandas")
    
    # Define optimized dtypes
    dtype_dict = {
        'id': 'object',
        'click': 'int8',
        'hour': 'object',  # Will convert after reading
        'C1': 'int16',
        'banner_pos': 'int8',
        'site_id': 'category',
        'site_domain': 'category',
        'site_category': 'category',
        'app_id': 'category',
        'app_domain': 'category',
        'app_category': 'category',
        'device_id': 'category',
        'device_ip': 'category',
        'device_model': 'category',
        'device_type': 'int8',
        'device_conn_type': 'int8',
        'C14': 'int16',
        'C15': 'int16',
        'C16': 'int16',
        'C17': 'int16',
        'C18': 'int16',
        'C19': 'int16',
        'C20': 'int16',
        'C21': 'int16'
    }
    
    # Read CSV
    if use_cudf:
        df = df_lib.read_csv(csv_path)
        # Convert hour column for cuDF
        if 'hour' in df.columns:
            try:
                # Try to convert hour to integer (assumes YYYYMMDDHH format)
                df['hour'] = df['hour'].astype('int32')
            except Exception:
                # If string format, extract hour
                df['hour'] = df_lib.to_datetime(df['hour']).dt.hour
    else:
        df = df_lib.read_csv(csv_path, dtype=dtype_dict)
        # Convert hour column for pandas
        if 'hour' in df.columns and df['hour'].dtype == 'object':
            try:
                # Try to convert hour to integer (assumes YYYYMMDDHH format)
                df['hour'] = pd.to_numeric(df['hour'], errors='coerce').fillna(0).astype('int32')
            except Exception:
                # If string format, extract hour
                df['hour'] = pd.to_datetime(df['hour']).dt.hour
    
    # Sample data if requested
    if sample_frac and sample_frac < 1.0:
        if use_cudf:
            df = df.sample(frac=sample_frac, random_state=42)
        else:
            df = df.sample(frac=sample_frac, random_state=42)
        print(f"📊 Sampled {sample_frac:.1%} of data")
    
    print(f"✅ Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"📋 Columns: {list(df.columns)}")
    
    return df