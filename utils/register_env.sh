#!/usr/bin/env bash
set -e
source .env
az configure --defaults group="$RESOURCE_GROUP" workspace="$WORKSPACE_NAME"

# Create or update the environment
az ml environment create \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE_NAME" \
  --file "env/environment.yml"