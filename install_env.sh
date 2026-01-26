#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

#######################################
# Start
#######################################
echo "Starting environment installation and dataset preparation..."

#########################################
# Install system packages and Python requirements
#########################################
./shells/install_requirements.sh

#########################################
# Download datasets
#########################################
./shells/download_unzip_dataset.sh

########################################
# Create dataset symbolic links
########################################
./shells/create_dataset_link.sh

#######################################
# Other assets
#######################################
./shells/prepare_other_asset.sh

#######################################
# Done
#######################################
echo "All packages installed and datasets downloaded successfully."