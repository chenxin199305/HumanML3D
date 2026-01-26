#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

#######################################
# 根据输入决定是否执行本脚本
#######################################
read -p "This script will prepare other necessary assets. Do you want to continue? (y/n) " choice
case "$choice" in
  y|Y ) echo "Proceeding with preparing other assets...";;
  n|N ) echo "Preparing other assets aborted."; exit 0;;
  * ) echo "Invalid input. Please enter 'y' or 'n'."; exit 1;;
esac
echo ""

#######################################
# Other assets
#######################################
echo "Unzipping the humanact12 model..."
HUMANACT12_ZIP="pose_data/humanact12.zip"
HUMANACT12_UNZIP_PARENT_DIR="pose_data"
HUMANACT12_UNZIP_DIR="pose_data/humanact12"

if [ ! -d "$HUMANACT12_UNZIP_DIR" ]; then
    unzip $HUMANACT12_ZIP -d $HUMANACT12_UNZIP_PARENT_DIR
else
    echo "humanact12 model already unzipped."
fi
echo ""

# Create folder named "joints"
echo "Creating folder named 'joints'..."
JOINTS_DIR="joints"

if [ ! -d "$JOINTS_DIR" ]; then
    mkdir -p $JOINTS_DIR
else
    echo "'joints' folder already exists."
fi
echo ""
