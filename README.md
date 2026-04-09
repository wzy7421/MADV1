# MAD-QWT: Informatics-Enhanced Perception in Autonomous Driving

This repository includes the official PyTorch implementation for performing multimodal inference and training with the **MAD framework**, as detailed in our paper:  
*"Integrating Multi-Q Gabor Wavelet and Semantic Modeling for Informatics-Enhanced Perception in Autonomous Driving".*  
**MAD** is a robust drivable area segmentation and occupancy-aware risk assessment model that systematically couples monocular vision with Q-factor Wavelet Transform (QWT)-enhanced vibration signals to maintain stability under complex and degraded visual conditions.

---

## Requirements

- **Python 3.8+**
- **PyTorch 1.10+** (CUDA supported)
- **OpenCV**
- **SciPy**
- **PyWavelets** (for QWT processing)

Install all dependencies via:

```bash
pip install -r requirements.txt




Usage
1. Data Preparation

Please download the original SpaceNet, DeepGlobe, and Cityscapes datasets from their official sources and place them into the data/ directory.

Run the preprocessing script to format the labels for drivable area extraction:

python src/data_utils.py --dataset_path ./data/
