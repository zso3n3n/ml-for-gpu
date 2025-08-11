import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
load_dotenv()

def resolve_abfss(asset_name: str, version: str|int):
    mlc = MLClient(
        DefaultAzureCredential(),
        os.getenv("AZ_SUBSCRIPTION_ID"),
        os.getenv("AZ_RESOURCE_GROUP"),
        os.getenv("AZ_ML_WORKSPACE"),
    )
    ds_name = os.getenv("AZ_DATASTORE")
    asset = mlc.data.get(name=asset_name, version=str(version))
    ds = mlc.datastores.get(ds_name)
    account, container = ds.account_name, ds.container_name
    prefix = asset.path.split("/paths/", 1)[1].strip("/")
    return f"abfss://{container}@{account}.dfs.core.windows.net/{prefix}", account

def storage_options(account_name: str):
    return {"account_name": account_name, "credential": DefaultAzureCredential()}
