#!/bin/bash

# Resolve absolute path to the project root
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [ -z "$1" ]; then
    echo "Error: Please provide a run ID (e.g., ./scripts/clone_case.sh run_001)"
    exit 1
fi

RUN_ID=$1
SOURCE_DIR="$PROJECT_ROOT/openfoam/template"
TARGET_DIR="$PROJECT_ROOT/openfoam/runs/$RUN_ID"

echo "Cloning template into openfoam/runs/$RUN_ID..."
cp -r $SOURCE_DIR $TARGET_DIR

echo "Success! Case $RUN_ID is ready to run."