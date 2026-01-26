#!/bin/bash

set -e

#######################################
# 根据输入决定是否执行本脚本
#######################################
read -p "This script will download and unzip necessary datasets. Do you want to continue? (y/n) " choice
case "$choice" in
  y|Y ) echo "Proceeding with dataset download...";;
  n|N ) echo "Dataset download aborted."; exit 0;;
  * ) echo "Invalid input. Please enter 'y' or 'n'."; exit 1;;
esac
echo ""

#######################################
# Dataset directory handling
#######################################
DOWNLOAD_DIR="download"

#######################################
# Download datasets
#######################################
# Download amass dataset zip file if it doesn't exist
# Jason 2026-01-26: AMASS here use SMPL-H model version.
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

#######################################
# Validate downloads
#######################################
if [ -f "$AMASS_ZIP" ] && [ -f "$DMPLS_ZIP" ] && [ -f "$SMPL_ZIP" ]; then
    echo "All necessary files downloaded successfully."
else
    echo "Some files are missing. Please check the download process."
    exit 1
fi
echo ""

#######################################
# Unpack datasets
#######################################
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
