#!/usr/bin/env bash
# Run from the root of the repository

set -e

source .env
az account set --subscription $SUBSCRIPTION_ID
az configure --defaults group=$RESOURCE_GROUP workspace=$WORKSPACE_NAME

LOCAL_FILE="$1"
FILE_NAME=$(basename "$LOCAL_FILE")
ASSET_NAME="${FILE_NAME%.*}"
FILE_DESCRIPTION="$2"

echo "Uploading $LOCAL_FILE to Azure ML..."
echo "Name: $ASSET_NAME ($FILE_NAME)"
echo "Description: $FILE_DESCRIPTION"

# Get the storage account and container details for the datastore
# STORAGE_ACCOUNT=$(az ml datastore show --name $AZUREML_DATASTORE --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME --subscription $SUBSCRIPTION_ID --query "account_name" -o tsv)
CONTAINER_NAME=$(az ml datastore show --name $AZUREML_DATASTORE --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME --subscription $SUBSCRIPTION_ID --query "container_name" -o tsv)

# Use azcopy to upload the file
echo "Generating SaS token"
exp_date=$(date -u -d "+10 days" +%Y-%m-%dT%H:%MZ)
sas=$(az storage blob generate-sas \
  --account-name $STORAGE_ACCOUNT \
  --container-name $CONTAINER_NAME \
  --name $FILE_NAME \
  --permissions cw \
  --expiry $exp_date \
  -o tsv)

echo "Uploading..."
echo "Source: $LOCAL_FILE"
echo "Destination: https://${STORAGE_ACCOUNT}.blob.core.windows.net/${CONTAINER_NAME}/${FILE_NAME}"

azcopy copy \
  "$LOCAL_FILE" \
  "https://${STORAGE_ACCOUNT}.blob.core.windows.net/${CONTAINER_NAME}/${FILE_NAME}?${sas}" \
  --overwrite=true

# Register the uploaded blob as an Azure ML dataset
echo "Creating Data Asset..."

az ml data create \
  --name        $ASSET_NAME \
  --resource-group $RESOURCE_GROUP \
  --workspace   $WORKSPACE_NAME \
  --subscription $SUBSCRIPTION_ID \
  --type        uri_file \
  --path "azureml://datastores/${AZUREML_DATASTORE}/paths/${FILE_NAME}" \
  --description "$FILE_DESCRIPTION"