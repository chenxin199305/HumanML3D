#!/bin/bash

# 脚本运行必须在项目根目录下

set -e  # Exit immediately if a command exits with a non-zero status

#######################################
# 根据输入决定是否执行本脚本
#######################################
read -p "This script will create symbolic links to existing datasets. Do you want to continue? (y/n) " choice
case "$choice" in
  y|Y ) echo "Proceeding with creating symbolic links...";;
  n|N ) echo "Creating symbolic links aborted."; exit 0;;
  * ) echo "Invalid input. Please enter 'y' or 'n'."; exit 1;;
esac
echo ""

#######################################
# Parse arguments
#######################################
AMASS_PATH=""
DMPLS_PATH=""
SMPL_PATH=""
HUMANACT12_PATH=""

# 让用户输入各数据集的路径
read -p "Enter the path to the AMASS dataset (or leave blank to skip): " AMASS_PATH
read -p "Enter the path to the DMPLS model (or leave blank to skip): " DMPLS_PATH
read -p "Enter the path to the SMPL model (or leave blank to skip): " SMPL_PATH
echo ""

#######################################
# Helper function
#######################################
link_dataset () {
    local SRC="$1"
    local DST="$2"

    if [[ -z "$SRC" ]]; then
        return
    fi

    if [[ ! -d "$SRC" ]]; then
        echo "Error: source path does not exist: $SRC"
        exit 1
    fi

    mkdir -p "$(dirname "$DST")"

    if [[ -e "$DST" ]]; then
        if [[ -L "$DST" ]]; then
            echo "Symlink already exists: $DST -> $(readlink "$DST")"
        else
            echo "Error: $DST exists and is not a symlink."
            exit 1
        fi
    else
        ln -s "$SRC" "$DST"
        echo "Created symlink: $DST -> $SRC"
    fi
}

#######################################
# Create symlinks
#######################################
echo "Linking existing datasets..."

link_dataset "$AMASS_PATH"      "motion_data/amass_data"
link_dataset "$DMPLS_PATH"      "body_model/dmpls"
link_dataset "$SMPL_PATH"       "body_model/smplh"

echo ""
echo "All specified datasets linked successfully."
