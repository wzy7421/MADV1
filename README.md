# MAD: Informatics-Enhanced Perception in Autonomous Driving

This repository contains the official PyTorch implementation of the paper: **"Integrating Multi-Q Gabor Wavelet and Semantic Modeling for Informatics-Enhanced Perception in Autonomous Driving"**.

## 1. Project Structure
- `main.py`: Main script for training and evaluation.
- `qwt_core.py`: Implementation of the Multi-Q Gabor Wavelet Transform (QWT) and SRAP/MinIP projections.
- `data_utils.py`: Scripts for dataset loading and preprocessing.
- `models/`: Directory for storing pre-trained weights.
- `images/`: Sample images for quick testing.

## 2. Environment Setup
Create a virtual environment and install the required dependencies:
```bash
conda create -n mad_env python=3.8
conda activate mad_env
pip install -r requirements.txt