#!/usr/bin/env bash

th download_mnist.lua
mkdir -p th-mnist
echo "Generating datasets in torch format"
th gen_datasets.lua
mkdir -p npy-mnist
echo "Converting to numpy format"
python th2npy.py
