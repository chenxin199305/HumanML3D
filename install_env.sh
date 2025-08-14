#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Update the package list
echo "Updating package list..."
sudo apt-get update -y
echo ""

# Install necessary packages
echo "Installing necessary packages..."
sudo apt-get install -y \
      libboost-all-dev \
      build-essential

# Upgrade pip to the latest version
echo "Upgrading pip to the latest version..."
pip install --upgrade pip
echo ""

# Check conda installation
if command -v conda &> /dev/null; then
    echo "Conda is installed."
else
    echo "Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Check already in conda environment
if [[ -n "$CONDA_PREFIX" ]]; then
    echo "You are already in a conda environment: $CONDA_PREFIX"
else
    echo "Not in a conda environment. Exit."
    exit 1
fi

# Install the required packages
echo "Installing required packages..."
pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
echo ""

# Install packages from urls
pip install git+https://github.com/nghorbani/body_visualizer.git
pip install git+https://github.com/MPI-IS/configer
pip install git+https://github.com/MPI-IS/mesh.git

# Create download folder if it doesn't exist
DOWNLOAD_DIR="download"
mkdir -p $DOWNLOAD_DIR

# Download amass dataset zip file if it doesn't exist
echo "Downloading the amass dataset..."
AMASS_URL="https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/humanml3d/amass_data.zip"
AMASS_ZIP="$DOWNLOAD_DIR/amass_data.zip"

if [ ! -f "$AMASS_ZIP" ]; then
    wget -P $DOWNLOAD_DIR $AMASS_URL
else
    echo "AMASS dataset zip file already exists."
fi
echo ""

# Download DMPLS model if it doesn't exist
echo "Downloading the DMPLS model..."
DMPLS_URL="https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/DMPL/dmpls.zip"
DMPLS_ZIP="$DOWNLOAD_DIR/dmpls.zip"

if [ ! -f "$DMPLS_ZIP" ]; then
    wget -P $DOWNLOAD_DIR $DMPLS_URL
else
    echo "DMPLS model zip file already exists."
fi
echo ""

# Donwload SMPL model if it doesn't exist
echo "Downloading the SMPL model..."
SMPL_URL="https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/SMPL/smplh.zip"
SMPL_ZIP="$DOWNLOAD_DIR/smplh.zip"

if [ ! -f "$SMPL_ZIP" ]; then
    wget -P $DOWNLOAD_DIR $SMPL_URL
else
    echo "SMPL model zip file already exists."
fi
echo ""

# Check all necessary download files exist
if [ -f "$AMASS_ZIP" ] && [ -f "$DMPLS_ZIP" ] && [ -f "$SMPL_ZIP" ]; then
    echo "All necessary files downloaded successfully."
else
    echo "Some files are missing. Please check the download process."
    exit 1
fi
echo ""

# Unzip the AMASS dataset
echo "Unzipping the AMASS dataset..."
AMASS_UNZIP_PARENT_DIR="motion_data"
AMASS_UNZIP_DIR="motion_data/amass_data"

if [ ! -d "$AMASS_UNZIP_DIR" ]; then
    unzip $AMASS_ZIP -d $AMASS_UNZIP_PARENT_DIR
else
    echo "AMASS dataset already unzipped."
fi
echo ""

# Unpack the DMPLS model
echo "Unpacking the DMPLS model..."
DMPLS_UNZIP_PARENT_DIR="body_model"
DMPLS_UNZIP_DIR="body_model/dmpls"

if [ ! -d "$DMPLS_UNZIP_DIR" ]; then
    unzip $DMPLS_ZIP -d $DMPLS_UNZIP_PARENT_DIR
else
    echo "DMPLS model already unpacked."
fi
echo ""

# Unpack the SMPL model
echo "Unpacking the SMPL model..."
SMPL_UNZIP_PARENT_DIR="body_model"
SMPL_UNZIP_DIR="body_model/smplh"

if [ ! -d "$SMPL_UNPACK_DIR" ]; then
    unzip $SMPL_ZIP -d $SMPL_UNZIP_PARENT_DIR
else
    echo "SMPL model already unpacked."
fi
echo ""

# Success message
echo "All packages installed and datasets downloaded successfully."