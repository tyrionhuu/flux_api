#!/bin/bash
set -e

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate img2img

# Execute the start_api_service.py with all arguments
exec /opt/conda/envs/img2img/bin/python start_api_service.py "$@"
