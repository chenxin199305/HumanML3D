#!/bin/bash

set -e

#######################################
# 根据输入决定是否执行本脚本
#######################################
read -p "This script will install system packages and Python requirements. Do you want to continue? (y/n) " choice
case "$choice" in
  y|Y ) echo "Proceeding with installation...";;
  n|N ) echo "Installation aborted."; exit 0;;
  * ) echo "Invalid input. Please enter 'y' or 'n'."; exit 1;;
esac
echo ""


#######################################
# Update system packages
#######################################
echo "Updating package list..."
sudo apt-get update -y
echo ""

echo "Installing necessary packages..."
sudo apt-get install -y \
      libboost-all-dev \
      build-essential
echo ""

#######################################
# Python / Conda checks
#######################################
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

#######################################
# Install Python requirements
#######################################
echo "Installing required packages..."
pip install -r requirements.txt
echo ""

# Install packages from urls
#pip install git+https://github.com/nghorbani/body_visualizer.git
#pip install git+https://github.com/MPI-IS/configer
#pip install git+https://github.com/MPI-IS/mesh.git
